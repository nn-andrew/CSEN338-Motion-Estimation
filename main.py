import cv2

class MotionCompensation:

    def __init__(self, start_frame, file):
        self.start_frame = start_frame
        self.file = file

        self.cap = cv2.VideoCapture(file)

        # Get framerate, set delay per frame
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.delay = int(1000 / self.fps)

        # self.frames = [self.start_frame, self.start_frame + 1]

        self.generate_prediction_frames()

    def generate_prediction_frames(self):
        ret0, frame0 = self.get_frame(self.start_frame)
        ret1, frame1 = self.get_frame(self.start_frame + 1)

        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        search_window_width = 64
        search_window_height = 64
        block_width = 24
        block_height = 24

        print(frame_width)
        print('start')

        min_ssd = float('inf')
        min_pos = [0, 0]

        block = frame0[10:10+24, 10:10+24]
        search_window = frame1[0:0+64, 0:0+64]
        for i in range(0, 64 - 24 + 1, 4):
            for j in range(0, 64 - 24 + 1, 4):
                candidate = search_window[i:i+24, j:j+24]
                # ssd of block and current candidate
                curr_ssd = ((candidate - block) ** 2).sum()
                print(curr_ssd)

                if curr_ssd < min_ssd:
                    min_ssd = curr_ssd
                    min_pos = [i, j]

                self.playback(self.start_frame, self.start_frame, rect0=((i, j), (i + 24, j + 24)), rect1=((min_pos[0], min_pos[1]), (min_pos[0] + 24, min_pos[1] + 24)))

        print('best is', min_pos, 'with ssd =', min_ssd)

        
        # new_frame = frame0.copy()
        # new_frame[pelx, pely] = [0, 0, 255]
        # print(pelx,pely)
        # cv2.imshow('Frame', new_frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        print('done')

        self.cap.release()
        cv2.destroyAllWindows()

    def get_frame(self, frame_index):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        return self.cap.read()

    def playback(self, start_frame_index, end_frame_index, rect0: tuple[tuple[int, int], tuple[int, int]]=None, rect1: tuple[tuple[int, int], tuple[int, int]]=None):
        print(rect0)
        i = start_frame_index

        # Set starting frame
        # self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
        # ret, frame = self.cap.read()
        # if ret:
        #     cv2.imshow('Frame', frame)
        # i += 1

        while i <= end_frame_index:

            # Set frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()

            # If frame is unreadable, break
            if not ret:
                break

            # b,g,r = frame[0, 0]
            # quantized_frame = frame & 0b10000000

            cv2.putText(frame, str(i), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            if rect1:
                cv2.rectangle(frame, rect1[0], rect1[1], (0, 255, 0), 2)
            if rect0:
                cv2.rectangle(frame, rect0[0], rect0[1], (0, 0, 255), 2)
            # Show frame
            cv2.imshow('Frame', frame)
            # i = (i + 1) % len(self.frames)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            i += 1

        # self.cap.release()
        # cv2.destroyAllWindows()

x = MotionCompensation(20, './test_clip.mp4')

