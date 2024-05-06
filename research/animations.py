from manim import *
from grfgnn import NormalRobotGraph


class CreateURDF(Scene):

    def construct(self):

        # Load the A1 urdf
        path_to_urdf = Path(Path('.').parent, 'urdf_files', 'A1',
                            'a1.urdf').absolute()
        A1_URDF = NormalRobotGraph(path_to_urdf, 'package://a1_description/',
                                  'unitree_ros/robots/a1_description', True)

        # Create a Circle and give it text
        text = Text('base').scale(2)
        rect_1 = RoundedRectangle(corner_radius=0.5)

        self.play(Create(rect_1))
        self.play(Write(text))
