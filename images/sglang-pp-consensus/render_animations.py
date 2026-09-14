#!/usr/bin/env python3
"""Render original PP teaching GIFs and static step sheets. Requires Pillow >= 10.

Run: python render_animations.py [--font /path/to/CJK-font.ttc]
Output stays beside this script. No network, model runtime, or source-tree edits.
Frame durations are reading time, never measurements of SGLang execution.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

from PIL import Image, ImageDraw, ImageFont

W, H = 1000, 760
BG, INK, MUTED = '#f2f5fa', '#162b45', '#52647c'
BLUE, GREEN, AMBER, RED = '#245bdb', '#11775e', '#986000', '#b43845'
PALE = {BLUE: '#eaf0ff', GREEN: '#e7f4ef', AMBER: '#fff3d9', RED: '#fcebed'}
X = [36, 362, 688]
OUT = Path(__file__).resolve().parent
FONT = None


def font(size):
    return ImageFont.truetype(FONT, size)


def text(d, xy, value, size=28, fill=INK):
    d.text(xy, value, font=font(size), fill=fill)


def fit_text(d, xy, value, width, size=28, fill=INK):
    assert d.textlength(value, font=font(size)) <= width, (value, width, size)
    text(d, xy, value, size, fill)


def box(d, coords, fill='white', outline=None, width=2, radius=18):
    d.rounded_rectangle(coords, radius, fill, outline, width)


def line_arrow(d, a, b, color=BLUE, width=4):
    d.line([a, b], fill=color, width=width)
    theta = math.atan2(b[1]-a[1], b[0]-a[0])
    p = [(b[0]-13*math.cos(theta+s), b[1]-13*math.sin(theta+s)) for s in [-.5,.5]]
    d.polygon([b,*p], fill=color)


# Each scene: short title, two explanation lines, three card rows, active cards,
# optional movement between two card centers, moving label, conclusion.
SCENARIOS = [
 dict(stem='06-rid-consensus', title='共识名单怎样逐站汇总？',
      subtitle='三个工位看同一批订单：全部 ready 才能进入 good',
      cards=['PP0 · 工位 0','PP1 · 工位 1','PP2 · 工位 2'],
      foot=['只演正向候选归约；不是多数票。', '名单回传、实际出队和资源检查，见正文第 5～7 节。'],
      scenes=[
       ('01  先看各站的本地判断', ['同一张订单，各站的准备情况可能不同。','绿色是 good 候选，红色是 bad 候选。'],
        [('good: A B C','bad: D'),('good: A B','bad: C'),('good: A C','bad: E')], [], None, '', '先收集候选，还没有决定谁可以运行。'),
       ('02  PP0 发出第一份候选', ['从 PP0 开始，把已有候选送到 PP1。','包里装的是 RID 名单，不是 KV 或激活 tensor。'],
        [('good: A B C','bad: D'),('good: A B','bad: C'),('good: A C','bad: E')], [0], (0,1), '候选', '累计 good = A B C    bad = D'),
       ('03  PP1 合并自己的判断', ['good 取交集：C 在这一站不 ready。','bad 取并集：把 C 的失败也带上。'],
        [('good: A B C','bad: D'),('good: A B','bad: C'),('good: A C','bad: E')], [1], None, '', '累计 good = A B    bad = C D'),
       ('04  把累计候选送到 PP2', ['继续传递汇总后的名单。','每一站都要参与，不能跳过尚未 ready 的工位。'],
        [('good: A B C','bad: D'),('good: A B','bad: C'),('good: A C','bad: E')], [1], (1,2), '候选', '累计 good = A B    bad = C D'),
       ('05  PP2 完成正向汇总', ['只有 A 在三站都属于 good。','B 没有全站 ready；C、D、E 出现在 bad 中。'],
        [('good: A B C','bad: D'),('good: A B','bad: C'),('good: A C','bad: E')], [2], None, '', '汇总 good = A    bad = C D E'),
       ('06  看懂结果，别多推一步', ['A：进入 good 候选结果；B：本轮继续等待。','C、D、E：进入 bad；这不代表已完成物理释放。'],
        [('A','good：状态资格'),('B','尚未全站 ready'),('C D E','bad：失败名单')], [0,1,2], None, '', '汇总名单 ≠ 所有 stage 已经实际出队。'),
      ]),
 dict(stem='07-admission-gates', title='已经 good，为什么还不能运行？',
      subtitle='跟着请求 A，看一个 Prefill stage 的本地准入',
      cards=['bootstrap queue','waiting queue','Prefill 计算'],
      foot=['单站、正常成功路径的教学示意；省略其他请求与 chunk。', '动画停顿用于阅读，不代表真实耗时或固定调度周期。'],
      scenes=[
       ('01  收到 A 的 good 名单', ['握手资格已经满足。','A 仍要在本地完成资源检查和 sender 初始化。'],
        [('A 在这里','收到 good'),('尚未进入','等待本地准入'),('尚未执行','等待调度')], [0], None, '', 'good 表示资格，不表示已经拿到所有资源。'),
       ('02  metadata 槽位暂时不足', ['finalize_bootstrap() 没能完成。','A 留在 bootstrap queue，不能直接跳到计算。'],
        [('A 继续等待','metadata 不足'),('尚未进入','没有实际出队'),('尚未执行','没有当前 batch')], [0], None, '', '像拿到开工通知，却暂时没有空工作台。'),
       ('03  资源满足，初始化成功', ['后续处理时拿到 metadata 槽位。','finalize 成功后，A 才能转入 waiting queue。'],
        [('A 完成准入','metadata 可用'),('接收请求 A','成功出队后'),('尚未执行','仍需调度')], [0,1], (0,1), 'A', '这次移动表示队列迁移，不是模型计算。'),
       ('04  A 在 waiting queue 排队', ['当前 batch 可能已经选好。','不能把刚刚准入的 A 倒灌进已选好的 batch。'],
        [('准入已完成','离开本队列'),('A 在这里','等后续调度'),('当前批已选好','A 尚未运行')], [1], None, '', 'waiting 是可供调度，不等于立刻执行。'),
       ('05  后续调度选中 A', ['满足调度条件并被选入后续 batch。','这时才进入本 stage 的 Prefill 计算。'],
        [('准入已完成','本地资源就绪'),('A 被选中','形成后续 batch'),('执行请求 A','本 stage 模型层')], [1,2], (1,2), 'A', '资格 → 实际准入 → 等待调度 → 执行。'),
      ]),
 dict(stem='08-abort-deferred-release', title='取消请求后，为什么还占着 KV？',
      subtitle='请求 A 退场，与槽位 S 交给请求 B，是两个动作',
      cards=['请求 A / 旧写入','KV 槽位 S','新请求 B'],
      foot=['只演满足延迟释放条件、后端判安全后释放的一条路径。', '另有超时释放分支；超时不等于旧写入已停止。'],
      scenes=[
       ('01  A 的传输还在进行', ['A 占用槽位 S，可能还有在途写入。','B 想使用资源，但现在不能复用这个槽位。'],
        [('A 正在传输','写入尚未退场'),('归 A 使用','尚未释放'),('B 等待资源','不能复用 S')], [0,1], (0,1), 'A 的数据', '订单号变化，不会自动停止已经发出的写入。'),
       ('02  A 被取消，进入延迟释放', ['业务上 A 已退出普通 transfer queue。','满足延迟释放条件时，仍保留目标 KV 与 metadata。'],
        [('业务已取消','旧写入需检查'),('继续为 A 保留','deferred hold'),('B 继续等待','S 不能复用')], [0,1], None, '', '业务取消 ≠ 物理资源立即可复用。'),
       ('03  后端尚未判安全', ['resolver 检查 release-safe，尚未安全且未到 deadline。','继续持有槽位，避免把旧写入风险交给 B。'],
        [('检查旧写入','尚未判安全'),('继续为 A 保留','尚未超时'),('B 继续等待','S 仍被保留')], [1], (0,1), '旧写入', '像取消订单后，仍要确认旧机械动作是否结束。'),
       ('04  后端报告可安全释放', ['本例走 release-safe 返回真的分支。','按后端约定满足条件后，调用本地释放。'],
        [('后端判安全','release-safe'),('执行本地释放','归还资源池'),('B 等待分配','尚未自动占用')], [0,1], None, '', '这里的依据是后端安全条件，不是动画等了几秒。'),
       ('05  槽位可供后续请求分配', ['资源归还后，后续调度才可能把 S 分配给 B。','这不是说 release-safe 会直接把槽位指定给 B。'],
        [('A 已完成清理','本地释放结束'),('S 已归还','可供后续分配'),('B 可尝试申请','仍由分配器决定')], [1,2], None, '', '先结束旧生命周期，再开始新一次使用。'),
      ]),
]


def draw_frame(spec, idx, progress=1):
    title, explain, rows, active, move, label, result = spec['scenes'][idx]
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    text(d,(36,22),'SGLang PP · 动画学习',22,BLUE)
    fit_text(d,(36,60),spec['title'],920,40)
    fit_text(d,(36,120),spec['subtitle'],920,27,MUTED)
    box(d,(36,174,964,234),INK)
    fit_text(d,(56,186),title,880,30,'white')
    headers = spec['cards']
    if spec['stem'] == '06-rid-consensus' and idx == 5:
        headers = ['A · 准入候选', 'B · 继续等待', 'C / D / E · bad']
    for i,(head,lines) in enumerate(zip(headers,rows)):
        color=BLUE if i in active else '#cad3e1'
        if spec['stem']=='06-rid-consensus' and idx==5:
            color=[GREEN,AMBER,RED][i]
        box(d,(X[i],263,X[i]+276,419),PALE.get(color,'white'),color,3)
        fit_text(d,(X[i]+16,278),head,244,25,BLUE)
        for j,value in enumerate(lines):
            tint=INK
            if spec['stem'].startswith('06') and idx<5:
                tint=GREEN if j==0 else RED
                box(d,(X[i]+10,320+42*j,X[i]+266,356+42*j),PALE[tint],radius=7)
            fit_text(d,(X[i]+16,323+42*j),value,244,27,tint)
    # A moving labelled token follows the logical handoff track, not physical time.
    if move:
        a,b=move;start=(X[a]+138,454);end=(X[b]+138,454)
        line_arrow(d,start,end,'#b7c4dc')
        q=progress*progress*(3-2*progress)
        cx=start[0]+(end[0]-start[0])*q
        width=max(76,int(d.textlength(label,font=font(24)))+28)
        box(d,(int(cx-width/2),434,int(cx+width/2),474),BLUE,radius=12)
        text(d,(int(cx-width/2)+14,437),label,24,'white')
    elif not (spec['stem']=='06-rid-consensus' and idx==5):
        for i in range(2):line_arrow(d,(X[i]+138,454),(X[i+1]+138,454),'#c3cddb',3)
    for j,value in enumerate(explain):fit_text(d,(40,499+j*39),value,920,27)
    box(d,(36,590,964,648),PALE[GREEN])
    fit_text(d,(52,601),result,896,26,GREEN)
    for j,value in enumerate(spec['foot']):fit_text(d,(38,666+j*31),value,922,23,MUTED)
    # Reading progress, explicitly a scene counter rather than elapsed runtime.
    for j in range(len(spec['scenes'])):
        x=820+j*24;d.ellipse((x,26,x+12,38),fill=BLUE if j==idx else '#ced8e8')
    return im


def storyboard(spec):
    n=len(spec['scenes']);im=Image.new('RGB',(1000,180+142*n),BG);d=ImageDraw.Draw(im)
    text(d,(36,24),spec['title']+' · 静态步骤',34)
    text(d,(36,78),'按序阅读；对应 GIF 的关键状态，不按真实时间比例绘制。',25,MUTED)
    for i,(title,explain,rows,active,move,label,result) in enumerate(spec['scenes']):
        y=136+i*142;box(d,(36,y,964,y+126))
        text(d,(54,y+10),title,28,BLUE)
        fit_text(d,(54,y+52),explain[0],892,25)
        fit_text(d,(54,y+88),result,892,24,GREEN)
    return im


def main():
    global FONT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font',type=Path)
    args=parser.parse_args()
    fonts=[args.font,Path('/System/Library/Fonts/STHeiti Medium.ttc'),
           Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
    FONT=next((str(p) for p in fonts if p and p.is_file()),None)
    if FONT is None:raise SystemExit('Provide --font with a Chinese-capable TTF/TTC font.')
    report=[]
    for spec in SCENARIOS:
        frames=[];durations=[];keyframes=[]
        for idx,scene in enumerate(spec['scenes']):
            keyframes.append(len(frames))
            # Hold each new scene before moving its token so labels can be read.
            frames.append(draw_frame(spec,idx,0 if scene[4] else 1));durations.append(2300)
            if scene[4]:
                for k in range(1,13):frames.append(draw_frame(spec,idx,k/12));durations.append(80)
                durations[-1]=1100
        durations[-1]+=1400
        # A shared palette avoids color flicker between GIF frames.
        # Pin semantic colors; sampling a resized swatch can wash out small text.
        base=[BG,INK,MUTED,BLUE,GREEN,AMBER,RED,*PALE.values(),
              '#ffffff','#cad3e1','#b7c4dc','#c3cddb','#ced8e8']
        colors=[tuple(int(c[k:k+2],16) for k in (1,3,5)) for c in base]
        for c in list(colors):
            for alpha in (.12,.25,.4,.55,.7,.85):
                colors.append(tuple(round(v*alpha+255*(1-alpha)) for v in c))
        for v in range(0,256,4):colors.append((v,v,v))
        colors=(colors+[(255,255,255)]*256)[:256]
        palette=Image.new('P',(1,1))
        palette.putpalette([v for color in colors for v in color])
        paletted=[frame.quantize(palette=palette,dither=Image.Dither.NONE) for frame in frames]
        gif=OUT/(spec['stem']+'.gif')
        paletted[0].save(gif,save_all=True,append_images=paletted[1:],duration=durations,loop=0,optimize=False,disposal=1)
        still=OUT/(spec['stem']+'-steps.png');storyboard(spec).save(still)
        with Image.open(gif) as saved:
            actual_durations=[]
            for j in range(saved.n_frames):saved.seek(j);actual_durations.append(saved.info['duration'])
            assert sum(actual_durations)==sum(durations)
            report.append(dict(file=gif.name,size=[W,H],frames=saved.n_frames,duration_ms=sum(actual_durations),
                loop=saved.info.get('loop'),bytes=gif.stat().st_size,sha256=hashlib.sha256(gif.read_bytes()).hexdigest(),
                scene_start_ms=[sum(durations[:k]) for k in keyframes],static=still.name,
                static_sha256=hashlib.sha256(still.read_bytes()).hexdigest()))
    (OUT/'animation-manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
