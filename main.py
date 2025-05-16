import cv2
import numpy as np

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
        ret1, frame1 = self.get_frame(self.start_frame + 1)

        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        search_window_width = 64
        search_window_height = 64
        anchor_block_width = 24
        anchor_block_height = 24
        candidate_block_width = anchor_block_width
        candidate_block_height = anchor_block_height

        print(frame_width)
        print('start')

        min_ssd = None
        min_pos = None

        # self.motion_vectors = np.full((frame_height - search_window_height + 1, frame_width - search_window_width + 1, 2), np.nan)
        self.motion_vectors = {}

        # m represents center of anchor block
        for m in range(anchor_block_width//2, frame_width - anchor_block_width//2 + 1, 16):
            for n in range(anchor_block_height//2, frame_height - anchor_block_height//2 + 1, 16):
                min_ssd = float('inf')
                min_pos = [m, n]
                # search_window = frame1[m:m+search_window_width, n:n+search_window_height]
                anchor_block = frame1[m-anchor_block_width//2 : m+anchor_block_width//2, n-anchor_block_height//2:n+anchor_block_height//2] # TODO: center of search window?

                # i, j represents center of candidate block, scans the search window (absolute coordinates)
                for i in range(m - search_window_width//2, m + search_window_width//2 + 1, 16):
                    for j in range(n - search_window_height//2, n + search_window_height//2 + 1, 16):
                        if i - candidate_block_width//2 < 0 or i + candidate_block_width//2 >= frame_width or j - candidate_block_height//2 < 0 or j + candidate_block_height//2 >= frame_height:
                            continue
                        candidate_block = frame0[i-candidate_block_width//2:i+candidate_block_width//2, j-candidate_block_height//2:j+candidate_block_height//2]
                        # ssd of block and current candidate
                        curr_ssd = ((candidate_block - anchor_block) ** 2).sum()
                        # print(curr_ssd)

                        if curr_ssd < min_ssd:
                            min_ssd = curr_ssd
                            min_pos = [i, j]

                        # rect0: curr candidate block
                        # rect1: best candidate block
                        # rect2: search window
                        self.playback(
                            self.start_frame + 1, 
                            self.start_frame + 1, 
                            rect0=((i - candidate_block_width//2, j - candidate_block_height//2), (i + candidate_block_width//2, j + candidate_block_height//2)), 
                            rect1=((min_pos[0] - candidate_block_width//2, min_pos[1] - candidate_block_height//2), (min_pos[0] + candidate_block_width//2, min_pos[1] + candidate_block_height//2)),
                            rect2=((m - search_window_width//2, n - search_window_height//2), (m + search_window_width//2, n + search_window_height//2)),
                            arrow=((m, n), (min_pos[0], min_pos[1])),
                            show_motion_vectors=True
                        )
                print('best is', min_pos, 'with ssd =', min_ssd)

                self.motion_vectors[(m, n)] = (min_pos[0] - m, min_pos[1] - n)

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
                        x, y = key
                        x2, y2 = val

                        cv2.arrowedLine(frame, (x, y), (x + x2, y + y2), (0, 255, 0), 2, tipLength=0.1)

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

x = MotionCompensation(905, './doom.mp4')
x.generate_prediction_frames()
x.playback(905, 906, autoplay=False)

