from sailenv.agent import Agent
from PIL import Image
import numpy as np



if __name__ == '__main__':

    agent = Agent(depth_frame_active=False,
                  flow_frame_active=False,
                  object_frame_active=False,
                  main_frame_active=True,
                  category_frame_active=False,
                  width=224, height=224, host="localhost",
                  port=8085, use_gzip=False)
    agent.register()

    frame = agent.get_frame()

    img = frame['main'] * 255
    img2 = Image.fromarray(img.astype(np.uint8))

    img2.save("x.png")