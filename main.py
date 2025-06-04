import cv2
import numpy as np
from numba import njit

@njit
def compute_ssd(block1, block2):
    return ((block1 - block2) ** 2).mean()

class MotionCompensation:

    def __init__(self, start_frame, file):
        self.start_frame = start_frame
        self.file = file

        self.cap = cv2.VideoCapture(file)

        # Get framerate, set delay per frame
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.delay = int(1000 / self.fps)

        self.motion_vectors = None

    def generate_prediction_frames(self):
        ret0, frame0 = self.get_frame(self.start_frame)
        ret1, frame1 = self.get_frame(self.start_frame + 3)

        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cx, cy = frame_width // 2, frame_height // 2  # image center
        f = 220  # example focal length, adjust as needed

        search_window_width = 64
        search_window_height = 64
        anchor_block_width = 16
        anchor_block_height = 16
        candidate_block_width = anchor_block_width
        candidate_block_height = anchor_block_height

        print(frame_width)
        print('start')

        best_ssd = None
        best_pos = None
        self.motion_vectors = {}

        def equisolid_to_perspective_radius(r_e, f):
            return f * np.tan(2 * np.arcsin(r_e / (2 * f)))

        def perspective_to_equisolid_radius(r_p, f):
            return 2 * f * np.sin(0.5 * np.arctan(r_p / f))

        def project_block_coords_grid(ax, ay, B, cx, cy, f):
            x = np.arange(ax - B//2, ax + B//2)
            y = np.arange(ay - B//2, ay + B//2)
            u, v = np.meshgrid(x - cx, y - cy)

            re = np.sqrt(u**2 + v**2)
            theta = 2 * np.arcsin(np.clip(re / (2 * f), 0, 1))
            rp = f * np.tan(theta)
            phi = np.arctan2(v, u + 1e-9)

            xp = rp * np.cos(phi)
            yp = rp * np.sin(phi)

            return xp + cx, yp + cy  # Return absolute image coordinates

        def reproject_grid_to_equisolid(xp, yp, cx, cy, f):
            rp = np.sqrt((xp - cx)**2 + (yp - cy)**2)
            theta = np.arctan(rp / f)
            re = 2 * f * np.sin(theta / 2)

            phi = np.arctan2(yp - cy, xp - cx + 1e-9)
            xe = re * np.cos(phi) + cx
            ye = re * np.sin(phi) + cy
            return xe, ye
        
        def compute_ssd_warped(candidate_frame, anchor_block, x_coords, y_coords, prediction_frame=None):
            h, w = candidate_frame.shape[:2]
            x_coords = np.clip(x_coords, 0, w - 1).astype(np.float32)
            y_coords = np.clip(y_coords, 0, h - 1).astype(np.float32)

            sampled = cv2.remap(candidate_frame, x_coords, y_coords, interpolation=cv2.INTER_LINEAR)
            mask = ~np.isnan(sampled)
            diff = (sampled - anchor_block)[mask]
            
            # Optional visualization
            if prediction_frame is not None:
                # Round warped coordinates to nearest pixels
                x_int = np.round(x_coords).astype(int)
                y_int = np.round(y_coords).astype(int)

                for yi, xi in zip(y_int.ravel(), x_int.ravel()):
                    if 0 <= yi < prediction_frame.shape[0] and 0 <= xi < prediction_frame.shape[1]:
                        prediction_frame[yi, xi] = [0, 0, 255]  # Red in BGR

                cv2.imshow('Warped Anchor Block', prediction_frame)
                cv2.waitKey(1)

            return np.sum(diff ** 2)



        c = 0
        for m in range(anchor_block_height//2, frame_height - anchor_block_height//2 + 1, anchor_block_height):
            for n in range(anchor_block_width//2, frame_width - anchor_block_width//2 + 1, anchor_block_width):
                anchor_block = frame1[m-8:m+8, n-8:n+8]  # 16x16 block
                xp_grid, yp_grid = project_block_coords_grid(n, m, 16, cx, cy, f)
                curr_xp, curr_yp = 0, 0
                best_ssd = float('inf')
                best_pos = (0, 0)

                jump = 8
                while True:
                    candidate_offsets = [[0, 0], [0, jump], [jump, 0], [0, -jump], [-jump, 0]]
                    pre_offset = (curr_xp, curr_yp)

                    for dxp, dyp in candidate_offsets:
                        x_shifted = xp_grid + dxp
                        y_shifted = yp_grid + dyp

                        ssd = compute_ssd_warped(
                            frame0, anchor_block, x_shifted, y_shifted, 
                            # prediction_frame=frame1.copy()
                        )
                        if ssd < best_ssd:
                            best_ssd = ssd
                            best_pos = (dxp, dyp)

                    if jump == 1:
                        break
                    if np.isclose(curr_xp, pre_offset[0]) and np.isclose(curr_yp, pre_offset[1]):
                        jump //= 2

                    curr_xp, curr_yp = best_pos

                self.motion_vectors[(m, n)] = best_pos

        prediction_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        for m in range(anchor_block_height//2, frame_height - anchor_block_height//2 + 1, anchor_block_height):
            for n in range(anchor_block_width//2, frame_width - anchor_block_width//2 + 1, anchor_block_width):
                dy, dx = self.motion_vectors[(m, n)]

                # Step 1: Warp entire anchor block to perspective domain
                xp_grid, yp_grid = project_block_coords_grid(n, m, anchor_block_width, cx, cy, f)

                # Step 2: Add motion vector in perspective domain
                xp_grid_shifted = xp_grid + dx
                yp_grid_shifted = yp_grid + dy

                xe, ye = reproject_grid_to_equisolid(xp_grid_shifted, yp_grid_shifted, cx, cy, f)

                warped_block = cv2.remap(
                    frame0, xe.astype(np.float32), ye.astype(np.float32),
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )

                # # Step 3: Sample from the reference frame using bilinear interpolation
                # x_sample = xp_grid_shifted.astype(np.float32)
                # y_sample = yp_grid_shifted.astype(np.float32)

                # warped_block = cv2.remap(
                #     frame0, x_sample, y_sample,
                #     interpolation=cv2.INTER_LINEAR,
                #     borderMode=cv2.BORDER_CONSTANT,
                #     borderValue=0
                # )

                # Step 4: Copy the block into the prediction frame
                prediction_frame[m - anchor_block_height//2 : m + anchor_block_height//2,
                                n - anchor_block_width//2 : n + anchor_block_width//2] = warped_block

        cv2.imshow('Blank Frame', prediction_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print('done')
        self.cap.release()
        cv2.destroyAllWindows()


    def get_frame(self, frame_index):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        return self.cap.read()

    def playback(self, 
                 start_frame_index, 
                 end_frame_index, 
                 autoplay=True,
                 rect0: tuple[tuple[int, int], tuple[int, int]]=None, 
                 rect1: tuple[tuple[int, int], tuple[int, int]]=None, 
                 rect2: tuple[tuple[int, int], tuple[int, int]]=None,
                 arrow: tuple[tuple[int, int], tuple[int, int]]=None,
                 show_motion_vectors: bool=False
                 ):

        wait_key_delay = 1
        if not autoplay:
            wait_key_delay = 0

        i = start_frame_index

        while i <= end_frame_index:

            # Set frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()

            # If frame is unreadable, break
            if not ret:
                break

            downsample_factor = 4
            c = 0
            if self.motion_vectors is not None:
                for key, val in self.motion_vectors.items():
                    if c % downsample_factor == 0:
                        y, x = key
                        y2, x2 = val

                        cv2.arrowedLine(frame, (x + x2, y + y2), (x, y), (0, 255, 0), 2, tipLength=0.1)

                    # Disabled to prevent downsampling
                    # c += 1

            if arrow:
                cv2.arrowedLine(frame, arrow[0], arrow[1], (0, 0, 255), 2, tipLength=0.1)
            if rect2:
                cv2.rectangle(frame, rect2[0], rect2[1], (255, 255, 0), 2)
            if rect1:
                cv2.rectangle(frame, rect1[0], rect1[1], (0, 255, 0), 2)
            if rect0:
                cv2.rectangle(frame, rect0[0], rect0[1], (0, 0, 255), 2)

            cv2.putText(frame, str(i), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            # Show frame
            cv2.imshow('Frame', frame)

            if cv2.waitKey(wait_key_delay) & 0xFF == ord('q'):
                break
            
            i += 1

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

i = 60
x = MotionCompensation(i, './skate.mp4')
x.generate_prediction_frames()
x.playback(i, i+1, autoplay=False)
