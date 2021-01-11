from sailenv.agent import Agent

host = "127.0.0.1"
if __name__ == '__main__':
    print("Generating agent...")
    agent = Agent(depth_frame_active=False,
                  flow_frame_active=False,
                  object_frame_active=False,
                  main_frame_active=False,
                  category_frame_active=False, width=256, height=192, host=host, port=8085, use_gzip=False)
    print("Registering agent on server...")
    agent.register()
    print(f"Agent registered with ID: {agent.id}")
    print(f"Available scenes: {agent.scenes}")
    # agent.change_scene("object_view/scene")

    # save_path = agent.send_obj_zip("./run.py")
    # print(f"File stored at {save_path}")

    agent.delete()