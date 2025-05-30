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

        # self.motion_vectors = np.full((frame_height - search_window_height + 1, frame_width - search_window_width + 1, 2), np.nan)
        self.motion_vectors = {}


        # generate motion vectors
        # m represents center of anchor block
        for m in range(anchor_block_height//2, frame_height - anchor_block_height//2 + 1, anchor_block_height):
            for n in range(anchor_block_width//2, frame_width - anchor_block_width//2 + 1, anchor_block_width):
                # search_window = frame1[m:m+search_window_width, n:n+search_window_height]
                anchor_block = frame1[m-anchor_block_height//2 : m+anchor_block_height//2, n-anchor_block_width//2:n+anchor_block_width//2]

                curr_y, curr_x = m, n

                candidate_block = frame0[curr_y - candidate_block_height//2 : curr_y + candidate_block_height//2, 
                                        curr_x - candidate_block_width//2 : curr_x + candidate_block_width//2]
                
                best_ssd = ((candidate_block - anchor_block) ** 2).sum()
                best_pos = [m, n]

                checked = set()
                
                jump = 8
                while True:
                    # Search candidate blocks above, below, left, right of anchor block position
                    candidate_offsets = [[0, 0], [0, jump], [jump, 0], [0, -jump], [-jump, 0]]

                    pre_offset_pos = [curr_y, curr_x]

                    for offset in candidate_offsets:
                        offset_y, offset_x = (curr_y + offset[0], curr_x + offset[1]) 
                        if (offset_y, offset_x) in checked:
                            continue

                        # Check if current candidate block is within bounds
                        if offset_x - candidate_block_width//2 < n - search_window_width//2 or offset_x + candidate_block_width//2 >= n + search_window_width//2:
                            continue
                        if offset_y - candidate_block_height//2 < m - search_window_height//2 or offset_y + candidate_block_height//2 >= m + search_window_height//2:
                            continue
                        if offset_x - candidate_block_width//2 < 0 or offset_x + candidate_block_width//2 >= frame_width:
                            continue
                        if offset_y - candidate_block_height//2 < 0 or offset_y + candidate_block_height//2 >= frame_height:
                            continue

                        # get ssd
                        candidate_block = frame0[offset_y - candidate_block_height//2 : offset_y + candidate_block_height//2, 
                                                offset_x - candidate_block_width//2 : offset_x + candidate_block_width//2]
                    
                        # compare ssd of current candidate block with best_ssd
                        offset_ssd = compute_ssd(candidate_block, anchor_block)
                        if offset_ssd < best_ssd:
                            best_ssd = offset_ssd
                            best_pos = [offset_y, offset_x]

                        checked.add((offset_y, offset_x))

                        # rect0: curr candidate block
                        # rect1: best candidate block
                        # rect2: search window
                        self.playback(
                            self.start_frame + 1, 
                            self.start_frame + 1, 
                            # autoplay=False,
                            rect0=((offset_x - candidate_block_width//2, offset_y - candidate_block_height//2), (offset_x + candidate_block_width//2, offset_y + candidate_block_height//2)), 
                            rect1=((best_pos[1] - candidate_block_width//2, best_pos[0] - candidate_block_height//2), (best_pos[1] + candidate_block_width//2, best_pos[0] + candidate_block_height//2)),
                            rect2=((n - search_window_width//2, m - search_window_height//2), (n + search_window_width//2, m + search_window_height//2)),
                            arrow=((best_pos[1], best_pos[0]), (n, m)),
                            show_motion_vectors=True
                        )
                    
                    if jump == 1:
                        break

                    # If, after checking all the offset candidate blocks, the best block hasn't changed, then end search
                    if best_pos == pre_offset_pos:
                        jump //=2

                    curr_y, curr_x = best_pos

                self.motion_vectors[(m, n)] = (best_pos[0] - m, best_pos[1] - n)

        # generate predicted frame
        prediction_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        for m in range(anchor_block_height//2, frame_height - anchor_block_height//2 + 1, anchor_block_height):
            for n in range(anchor_block_width//2, frame_width - anchor_block_width//2 + 1, anchor_block_width):
                mv_y, mv_x = self.motion_vectors[(m,n)]
                best_block = frame0[m + mv_y - candidate_block_height//2 : m + mv_y + candidate_block_height//2, 
                                        n + mv_x - candidate_block_width//2 : n + mv_x + candidate_block_width//2]
                prediction_frame[
                    m-anchor_block_height//2 : m+anchor_block_height//2, 
                    n-anchor_block_width//2:n+anchor_block_width//2
                ] = best_block

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
x = MotionCompensation(i, './wii.mp4')
x.generate_prediction_frames()
x.playback(i, i+1, autoplay=False)
