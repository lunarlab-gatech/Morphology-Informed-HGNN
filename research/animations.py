from manim import *
from manim.typing import Vector3D, Point3D
from grfgnn import NormalRobotGraph
from pathlib import Path

class CreateURDF(Scene):

    def node_type_to_color(self, node_type: str):
        if(node_type == 'base'):
            return ManimColor.from_hex('#D02536')
        elif(node_type == 'joint'):
            return ManimColor.from_hex('#F38C16')
        elif(node_type == 'foot'):
            return ManimColor.from_hex('#F4FF1F')
        else:
            raise ValueError

    def construct(self):
        # Load the A1 urdf
        path_to_urdf = Path(Path('.').parent, 'urdf_files', 'A1',
                            'a1.urdf').absolute()
        A1_URDF = NormalRobotGraph(path_to_urdf, 'package://a1_description/',
                                  'unitree_ros/robots/a1_description')
        
        # Create a rectangle for each node
        rectangles = []
        for i, node in enumerate(A1_URDF.nodes):
            # Make the rectangle
            color = self.node_type_to_color(node.get_node_type())
            rect = RoundedRectangle(corner_radius=0.2, color=color, height=1.0, width=1.5)
            rect.set_fill(color, opacity=0.5)

            # Add text to the rectangle
            text = Text(node.name).scale(0.25)
            rect.add(text)

            # Move the rectangle to the proper spot
            if i == 0:
                rect.move_to([0, 3, 0])
            else:
                i_div = int((i - 1) / 4)
                i_mod = (i - 1) % 4
                if i_mod == 0:
                    rect.move_to([2*(i_div-1.5), 1.5, 0])
                elif i_mod == 1:
                    rect.move_to([2*(i_div-1.5), 0, 0])
                elif i_mod == 2:
                    rect.move_to([2*(i_div-1.5), -1.5, 0])
                elif i_mod == 3:
                    rect.move_to([2*(i_div-1.5), -3, 0])

            # Add it to all the others
            rectangles.append(rect)
        
        # For each connection, make an arrow
        edge_matrix = A1_URDF.get_edge_index_matrix()
        arrows = []
        for j in range(0, edge_matrix.shape[1]):
            if j % 2 == 1:
                continue
            col = edge_matrix[:,j]

            # Get the two corresponding rectangles
            parent: RoundedRectangle = rectangles[col[0]]
            child: RoundedRectangle = rectangles[col[1]]

            # Make the arrow
            start = parent.get_center() + [0, -0.5, 0]
            end = child.get_center() + [0, 0.5, 0]
            arrow = Arrow(start, end, buff=0)
            arrows.append(arrow)

        
        # Play them on the screen
        rectangles_vgroup = VGroup(*rectangles)
        arrows = VGroup(*arrows)
        shift_vector = UP * 0.25
        self.play(FadeIn(rectangles_vgroup, shift=shift_vector, scale=0.9, run_time=1.0), FadeIn(arrows, shift=shift_vector, scale=0.9, run_time=1.0))
        self.wait(1)
        self.play(rectangles_vgroup.animate.shift(LEFT*2), arrows.animate.shift(LEFT*2))

        # Display the text "URDF File"
        right_side_placement = 4.5
        text_title = Text("URDF File", weight=BOLD, font="sans-serif").scale(1).move_to([right_side_placement, 3, 0])
        self.play(FadeIn(text_title, shift=shift_vector, scale=0.9, run_time=1.0))

        # Create three circles to classify the node types
        circle_base = Circle(color=self.node_type_to_color('base'), radius=0.25)
        circle_base.set_fill(self.node_type_to_color('base'), opacity=0.5)
        text = Text('base', slant=ITALIC).scale(0.4)
        text.next_to(circle_base, RIGHT, buff=0.3)
        circle_base.add(text)
        circle_base.move_to([right_side_placement, 1, 0])

        circle_joint = Circle(color=self.node_type_to_color('joint'), radius=0.25)
        circle_joint.set_fill(self.node_type_to_color('joint'), opacity=0.5)
        text = Text('joint', slant=ITALIC).scale(0.4)
        text.next_to(circle_joint, RIGHT, buff=0.3)
        circle_joint.add(text)
        circle_joint.move_to([right_side_placement, 0, 0])

        circle_foot = Circle(color=self.node_type_to_color('foot'), radius=0.25)
        circle_foot.set_fill(self.node_type_to_color('foot'), opacity=0.5)
        text = Text('foot', slant=ITALIC).scale(0.4)
        text.next_to(circle_foot, RIGHT, buff=0.3)
        circle_foot.add(text)
        circle_foot.move_to([right_side_placement, -1, 0])

        group = VGroup(*[circle_base, circle_joint, circle_foot])
        self.play(FadeIn(group, shift=shift_vector, scale=0.9, run_time=1.0))
        
        # Make most of the graph go away, but select one of each type to move to center
        base_rect = rectangles[0]
        joint_rect = rectangles[10]
        foot_rect = rectangles[8]

        text_new_title = Text("Node Representations", weight=BOLD, font="sans-serif").scale(1).move_to([-3,3,0])

        embeddings_rects = VGroup(base_rect, joint_rect, foot_rect)
        self.play(FadeOut(rectangles_vgroup-base_rect-joint_rect-foot_rect, shift=shift_vector, scale=0.9, run_time=1.0),
                  FadeOut(arrows, shift=shift_vector, scale=0.9, run_time=1.0),
                  embeddings_rects.animate.scale(1.5).arrange_in_grid(rows=3).move_to([-5,-0.5,0]),
                  ReplacementTransform(text_title, text_new_title))
        
        # Add text explaining which each one represents
        text_base = Text('The center of the robot with the IMU. \nData: [linear acceleration, \nangular velocity, angular acceleration]').scale(0.4)
        text_base.next_to(base_rect, RIGHT, buff=0.3)

        text_joint = Text('The joint motors on the quadruped legs. \nData: [position, velocity, \nacceleration, torque]').scale(0.4)
        text_joint.next_to(joint_rect, RIGHT, buff=0.3)

        text_foot = Text('The feet on the end-effectors. \nData: [ground reaction force]').scale(0.4)
        text_foot.next_to(foot_rect, RIGHT, buff=0.3)

        self.play(FadeIn(text_base), shift=shift_vector, scale=0.9, run_time=1.0)
        self.wait(2)
        self.play(FadeIn(text_joint), shift=shift_vector, scale=0.9, run_time=1.0)
        self.wait(2)
        self.play(FadeIn(text_foot), shift=shift_vector, scale=0.9, run_time=1.0)
        self.wait(2)

        text_base_new = MathTex('[a, \omega, \dot{\omega}]').scale(1)
        text_base_new.next_to(base_rect, RIGHT, buff=0.3)

        text_joint_new = MathTex('[x, \dot{x}, \ddot{x}, \\tau]').scale(1)
        text_joint_new.next_to(joint_rect, RIGHT, buff=0.3)

        text_foot_new = MathTex('[f_{z}]', ).scale(1)
        text_foot_new.next_to(foot_rect, RIGHT, buff=0.3)

        text_emb_title = Text("Node Inputs", weight=BOLD, font="sans-serif").scale(1).move_to([-4,3,0])

        self.play(ReplacementTransform(text_new_title, text_emb_title),
                  Transform(text_base, text_base_new),
                  Transform(text_joint, text_joint_new),
                  Transform(text_foot, text_foot_new))
        self.wait(2)
