from textwrap import dedent
from docutils import nodes
from docutils.parsers.rst import Directive


class x3d(nodes.Element):
    pass

class video_loop(nodes.Element):
    pass

class VideoLoopDirective(Directive):
    required_arguments = 1

    def run(self):
        node = video_loop()
        node['fname'] = self.arguments[0]
        return [node]

class X3DDirective(Directive):
    required_arguments = 1

    def run(self):
        node = x3d()
        node['fname'] = self.arguments[0]
        return [node]


def html_visit_x3d(self, node):
    html = dedent(f"""\
            <div style="width:500px; margin: auto">
                <x3d width='500px' height='400px'>
                    <scene>
                        <inline url={node['fname']}> </inline>
                    </scene>
                </x3d>
            </div>
            """)
    self.body.append(html)


def html_visit_vl(self, node):
    html = dedent(f"""\
            <div style="width:500px; margin: auto">
                <video width="500" autoplay loop muted playsinline >
                    <source src="_static/video/{node['fname']}.mp4" type="video/mp4">
                    <source src="_static/video/{node['fname']}.webm" type="video/webm">
                </video>
            </div>
            """)
    self.body.append(html)


def empty(self, node):
    pass


def setup(app):
    app.add_directive("x3d", X3DDirective)
    app.add_directive("video-loop", VideoLoopDirective)
    app.add_node(x3d,
                 html=(html_visit_x3d, empty),
                 latex=(empty, empty),
                 text=(empty, empty))
    app.add_node(video_loop,
                 html=(html_visit_vl, empty),
                 latex=(empty, empty),
                 text=(empty, empty))

    return {'version': '0.1',
            'parallel_read_safe': True,
            'parallel_write_safe': True}