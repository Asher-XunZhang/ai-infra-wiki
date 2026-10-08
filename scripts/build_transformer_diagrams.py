#!/usr/bin/env python3
"""Draw original Transformer teaching figures as data, geometry and examples.

Only standard library dependencies. --check detects generated SVG drift.
All numerical examples are synthetic, never measurements from model weights.
"""
import argparse
import math
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'images/transformer-architecture'
INK = '#192d42'
MUTED = '#536778'
BLUE = '#2870c5'
TEAL = '#168574'
PURPLE = '#8255b4'
ORANGE = '#c87917'
RED = '#c65255'
GRAY = '#a0adb9'
BG = '#fbfaf7'


def tint(color, amount=0.16):
    rgb = [int(color[i:i+2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(255*(1-amount)+v*amount):02x}' for v in rgb)


def fmt(v):
    if isinstance(v, str):
        return v
    return f'{v:g}'.replace('-', '−')


class Fig:
    def __init__(self, number, title, purpose, height=860, width=840):
        self.w, self.h = width, height
        self.p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
                  f'<title id="title">{number} · {escape(title)}</title><desc id="desc">{escape(purpose)}。人工教学示例，不代表实测模型。</desc>',
                  '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" fill="context-stroke"/></marker><pattern id="hatch" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M0,10 L10,0" stroke="#c65255" stroke-width="1" opacity=".3"/></pattern></defs>',
                  f'<rect width="{width}" height="{height}" fill="{BG}" rx="16"/>',
                  '<g font-family="PingFang SC, Noto Sans CJK SC, Microsoft YaHei, sans-serif">']
        self.text(32, 49, f'{number}  {title}', 32, bold=True)
        self.text(32, 90, purpose, 24, color=MUTED)
        self.line(32, 113, width-32, 113, '#d7ddd9')
        self.text(32, height-22, '原创教学示意 · 数字为人工例子 · ★ 核心模块', 19, color=MUTED)

    def text(self, x, y, value, size=26, color=INK, bold=False, anchor='start'):
        self.p.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{650 if bold else 400}" fill="{color}" text-anchor="{anchor}">{escape(str(value))}</text>')

    def lines(self, x, y, values, size=24, color=MUTED, step=35, **kw):
        for i, value in enumerate(values):
            self.text(x, y+i*step, value, size, color, **kw)

    def rect(self, x, y, w, h, fill='white', stroke='none', radius=6, dash=False):
        self.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"' + (' stroke-dasharray="6 5"' if dash else '') + '/>')

    def line(self, x1, y1, x2, y2, color=GRAY, width=2, arrow=False, dash=False, opacity=1):
        self.p.append(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity}"' + (' marker-end="url(#arrow)"' if arrow else '') + (' stroke-dasharray="6 5"' if dash else '') + '/>')

    def path(self, d, color=GRAY, width=2, arrow=False, dash=False, opacity=1):
        self.p.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"' + (' marker-end="url(#arrow)"' if arrow else '') + (' stroke-dasharray="7 5"' if dash else '') + '/>')

    def circle(self, x, y, r, color='white', stroke=GRAY, label=None, size=25):
        self.p.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" stroke="{stroke}" stroke-width="2"/>')
        if label is not None:
            self.text(x, y+size*.34, fmt(label), size, anchor='middle')

    def token(self, x, y, label, color=BLUE, w=108, pending=False):
        self.rect(x, y, w, 51, tint(color,.13) if not pending else BG, color, 10, pending)
        self.text(x+w/2, y+34, label, 26, color, True, 'middle')

    def vec(self, x, y, values, color=BLUE, cell=47, numeric=True, vertical=False):
        for i,v in enumerate(values):
            dx,dy = (0, i*cell) if vertical else (i*cell, 0)
            n = abs(v) if isinstance(v, (int,float)) else .4+i*.1
            fill=tint(color, .10+min(n/4,.65))
            self.rect(x+dx,y+dy,cell-3,cell-3,fill)
            if numeric:self.text(x+dx+(cell-3)/2,y+dy+cell*.67,fmt(v),23,anchor='middle')
        return (cell if vertical else len(values)*cell, len(values)*cell if vertical else cell)

    def grid(self,x,y,data,cell=58,color=TEAL,values=True,mask=False):
        for r,row in enumerate(data):
            for c,v in enumerate(row):
                blocked=v is None
                fill=tint(RED,.09) if blocked else tint(color,.08+min(abs(v),1)*.7)
                self.rect(x+c*cell,y+r*cell,cell-2,cell-2,fill,radius=3)
                if blocked and mask:self.rect(x+c*cell,y+r*cell,cell-2,cell-2,'url(#hatch)',radius=3)
                if values:self.text(x+c*cell+(cell-2)/2,y+r*cell+cell*.64,'×' if blocked else fmt(v),24,color=RED if blocked else INK,anchor='middle')

    def bar(self,x,y,w,value,color,label='',h=25):
        self.rect(x,y,w,h,tint(color,.07),radius=3)
        if value:self.rect(x,y,w*value,h,color,radius=3)
        if label:self.text(x+w+15,y+h*.8,label,24,color=color)

    def note(self, y, title, detail, color=TEAL):
        self.rect(32,y,self.w-64,100,tint(color,.07),radius=8)
        self.rect(32,y,5,100,color,radius=0)
        self.text(52,y+36,title,26,color=color,bold=True)
        self.text(52,y+74,detail,23)

    def save(self):
        return '\n'.join(self.p+['</g></svg>'])+'\n'


def token_lane(f, y, labels, start=180, spacing=168, color=BLUE):
    for i, label in enumerate(labels):
        f.token(start+i*spacing,y,label,color)


def mini_funnel(f,x,y,width=130,height=86,color=PURPLE):
    cols=[(x,[y+height*.3,y+height*.7]),(x+width/2,[y+i*height/4 for i in range(5)]),(x+width,[y+height*.3,y+height*.7])]
    for (xa,ys),(xb,zs) in zip(cols,cols[1:]):
        for a in ys:
            for b in zs:f.line(xa,a,xb,b,color,1,opacity=.25)
    for xx,ys in cols:
        for yy in ys:f.circle(xx,yy,4,color,'none')


def architecture_additions():
    out = {}
    f = Fig('23', 'Transformer 全模块总览 ★', '沿左右两条竖线向下读：源句先编码，目标前缀逐步生成；橙色长线把新 token 接回输入。', 2500, 1460)
    f.text(60, 168, '① 源句：先读懂', 32, BLUE, True)
    f.text(812, 168, '② 目标：接着写', 32, TEAL, True)
    f.text(60, 212, 'I love cats → Tokenizer → 源 IDs', 25)
    f.text(812, 212, '起始 BOS；此刻已有 BOS / 我', 25)
    for x, words, col in [(80, ['I', 'love', 'cats'], BLUE), (865, ['BOS', '我'], TEAL)]:
        for i, word in enumerate(words): f.token(x+i*151, 234, word, col, 124)
    def box(x, y, w, h, title, lines, color=BLUE):
        f.rect(x, y, w, h, tint(color, .08), color, 10)
        f.text(x+18, y+34, title, 26, color, True)
        f.lines(x+18, y+64, lines, 23, step=34)
    for x, c, w in [(90, BLUE, 460), (815, TEAL, 510)]:
        mid=x+w/2
        f.line(mid, 285, mid, 310, c, 3, True)
        box(x, 310, w, 80, '★ 词嵌入 Embedding', ['查 ID → 每个 token 一条 d 维数字带'], c)
        f.line(mid, 390, mid, 420, c, 3, True)
        box(x, 420, w, 80, '★ 加入位置信息', ['经典结构：缩放词嵌入后加正弦 PE'], c)
    f.rect(38, 532, 550, 756, 'white', BLUE, 16)
    f.rect(754, 532, 625, 1126, 'white', TEAL, 16)
    f.text(61, 573, 'Encoder 层 × Nₑ', 30, BLUE, True)
    f.text(815, 573, 'Decoder 层 × N_d', 30, TEAL, True)
    def sublayer(x, y, w, title, rows, color, mask=False, funnel=False):
        mid=x+w/2
        f.line(mid, y-30, mid, y, color, 3, True)
        box(x, y, w, 175, title, rows, color)
        if funnel: mini_funnel(f,x+w-110,y+106,75,43,color)
        # Every sublayer has its own input bypass and addition, before LN.
        bypass=x-27
        f.path(f'M{mid},{y-21} L{bypass},{y-21} L{bypass},{y+212} L{mid-18},{y+212}', ORANGE, 3, True)
        f.line(mid,y+175,mid,y+194,color,3,True)
        f.circle(mid,y+212,18,tint(ORANGE),ORANGE,'+',25)
        f.text(mid+30,y+220,'★ 残差相加',22,ORANGE)
        f.line(mid,y+230,mid,y+245,color,3,True)
        f.rect(x,y+245,w,52,tint(ORANGE,.07),ORANGE)
        f.text(mid,y+279,'★ LayerNorm：逐 token 整理尺度',24,ORANGE,True,'middle')
    for mid,col in [(320,BLUE),(1070,TEAL)]: f.line(mid,500,mid,584,col,3)
    sublayer(100, 610, 440, '★ 多头自注意力', ['Q / K / V 都来自本层源表示', '源位置互相读取 → 拼接 → W_O', '填充掩码：源 PAD 不可读'], BLUE)
    f.line(320,907,320,944,BLUE,3)
    sublayer(100, 970, 440, '★ FFN：每个位置各自加工', ['上投影 d → d_ff', '非线性激活 ReLU', '下投影 d_ff → d'], PURPLE, funnel=True)
    f.line(320,1267,320,1310,BLUE,3,True)
    box(60,1310,520,110,'最终源表示 H_enc：保留整排位置',['各层参数不同；最后一层输出供所有', 'Decoder 层的交叉注意力分别投影 K/V'],BLUE)
    f.path('M580,1365 L678,1365 L678,1074 L815,1074',BLUE,4,True)
    f.text(598,1290,'K / V',25,BLUE,True)
    sublayer(815,610,510,'★ 带因果掩码的多头自注意力',['Q / K / V 来自本层目标表示','前瞻掩码 + 目标填充掩码','只读取自身与合法历史 → 拼接 → W_O'],TEAL)
    f.line(1070,907,1070,944,TEAL,3)
    sublayer(815,970,510,'★ 交叉多头注意力',['Q：上一步目标表示；K/V：H_enc','源填充掩码：排除源 PAD','目标查询源句 → 拼接 → W_O'],BLUE)
    f.line(1070,1267,1070,1304,TEAL,3)
    sublayer(815,1330,510,'★ FFN：每个位置各自加工',['上投影 d → d_ff → 非线性 → 下投影 d','与左侧同类工序，使用自己的参数','现代变体：这个槽位可替换为 MoE'],PURPLE)
    f.line(1070,1627,1070,1690,TEAL,3,True)
    box(815,1690,510,110,'取最后有效目标位置 → LM Head',['d 维表示 → 词表 V 个分数 logits','本例：用“我”所在位置预测后继'],TEAL)
    f.line(1070,1800,1070,1830,TEAL,3,True)
    box(815,1830,510,110,'★ 解码选择：决定下一个 token',['Greedy 取最大分；或 Softmax 后采样','例如选出“喜欢”：此刻尚未处理它'],TEAL)
    f.line(1070,1940,1070,1970,TEAL,3,True)
    f.token(1008,1970,'喜欢',ORANGE,124,True)
    f.rect(793,2036,535,45,tint(RED,.05),RED,8)
    f.text(803,2068,'EOS / 长度等停止条件？',29,RED,True)
    f.line(1070,2021,1070,2036,ORANGE,3,True)
    f.path('M1328,2058 L1424,2058 L1424,259 L1170,259',ORANGE,4,True)
    f.text(1164,2027,'否：接回前缀',23,ORANGE)
    f.line(1070,2081,1070,2110,RED,3,True)
    f.text(1096,2102,'是',22,RED)
    box(815,2110,510,108,'结束：Token IDs → 输出文字',['去除协议特殊标记，再 Detokenize','最终文字示例：我喜欢猫'],RED)
    # Detail key: it is shared by all three attention sublayers above.
    box(60,1480,630,268,'放大任意一个注意力头',['① 本子层输入分别投影出 Q、K、V','② QKᵀ / √d_k：给“读哪个位置”打分','③ 加 mask：允许 +0；禁止 −∞','④ Softmax：每个 Query 的读取份额','⑤ 权重 × V：汇总内容','多个头分别算 → Concat 拼接 → W_O'],TEAL)
    box(60,1785,630,174,'MoE 放在哪里？见图 24–26',['替换层内 FFN：Router → Top-k 专家','选中专家各自做 FFN → 按权重相加','输出仍为 d 维；残差与归一化仍保留'],PURPLE)
    box(60,1990,630,230,'KV Cache 是推理复用，不是另一个子层',['Decoder 各层：旧 self-attention K/V 保留','新 token：计算并追加本层 K/V','当前 Q：仍读取全部合法 K/V','交叉注意力的源 K/V 也可逐层预先缓存','上图为完整前缀视图；缓存细节见图 21'],BLUE)
    f.note(2260,'读图边界：主干是经典 Encoder–Decoder / Post-LN','每层各有参数；框内所有子层随层堆叠重复。PAD 掩码来自输入有效长度。')
    f.lines(52,2396,['现代 Decoder-only：移除独立 Encoder 和交叉注意力，prompt 与输出共用一条序列。','Pre-Norm / RMSNorm、RoPE、SwiGLU 与 MoE 是需按模型选择的变体；对照图 22。'],25,step=39)
    out['23-full-architecture.svg']=f.save()

    f=Fig('24','MoE：同一层，多几个可选的加工台 ★','Attention 负责读上下文；MoE 在 FFN 的位置，为每个 token 选择加工网络。',1120,1000)
    f.text(45,162,'先走共同的注意力、残差与归一化',28,TEAL,True)
    for i,word in enumerate(['我','喜欢','猫']):f.token(149+i*282,235,word,BLUE,118)
    f.text(44,211,'普通 Dense FFN',29,PURPLE,True)
    for i in range(3):
        x=208+i*282
        f.line(x,286,x,329,BLUE,2,True)
        mini_funnel(f,x-63,348,126,64)
        f.text(x,458,'同一套 FFN 参数',24,PURPLE,anchor='middle')
    f.text(45,526,'换成 MoE：先分配，再各自加工',29,PURPLE,True)
    routes=[(0,1),(1,3),(2,3)]
    for i,word in enumerate(['我','喜欢','猫']):
        x=209+i*282
        f.token(x-59,558,word,BLUE,118)
        f.text(x,649,'Router 选两位',24,ORANGE,anchor='middle')
        for e in routes[i]:f.line(x,671,140+240*e,740,BLUE,2,True,opacity=.45)
    for i in range(4):
        x=140+i*240
        f.rect(x-91,752,182,160,tint(PURPLE,.09),PURPLE,12)
        f.text(x,790,f'专家 {i+1}',27,PURPLE,True,'middle')
        mini_funnel(f,x-52,816,104,50)
        f.text(x,893,'独立的 FFN 参数',20,anchor='middle')
    f.note(952,'每个 token 分别汇总自己的专家输出','图中各 token 选不同的两位；不是把三个 token 混成一个。',PURPLE)
    out['24-moe-placement.svg']=f.save()

    f=Fig('25','MoE 手算：选四位中的两位，再汇总 ★','同一个输入送入选中专家；汇总得到新的数字带，宽度仍是 d。',1370,1040)
    f.text(48,167,'① 当前 token 的表示 x',28,BLUE,True)
    f.vec(700,135,[1,2],BLUE,58)
    f.text(48,226,'Router 线性投影：d 维 → 4 个专家分数',26)
    f.text(48,276,'本例 logits = [0, ln 6, ln 2, 0]；Top-2 选专家 2、3',25)
    f.text(48,330,'② 只在选中的分数之间 Softmax',28,ORANGE,True)
    for i,(v,label) in enumerate([(0,'专家 1：不计算'),(.75,'专家 2：6 / (6+2) = 0.75'),(.25,'专家 3：2 / (6+2) = 0.25'),(0,'专家 4：不计算')]):
        y=364+i*49
        f.text(70,y+20,str(i+1),24)
        f.bar(120,y,370,v,PURPLE if v else GRAY,label)
    f.text(48,604,'③ 两位专家分别加工同一个 x',28,PURPLE,True)
    f.vec(462,638,[1,2],BLUE,49)
    for x,e,vals in [(261,2,[2,0]),(778,3,[0,4])]:
        f.line(510,700,x,759,BLUE,3,True)
        f.rect(x-181,775,362,217,tint(PURPLE,.09),PURPLE,13)
        f.text(x,817,f'专家 {e}：自己的 FFN',28,PURPLE,True,'middle')
        mini_funnel(f,x-66,848,132,59)
        f.vec(x-54,930,vals,PURPLE,54)
    f.text(48,1051,'④ 按路由权重汇总两个输出',28,TEAL,True)
    f.text(520,1110,'0.75 × [2, 0] + 0.25 × [0, 4] = [1.5, 1]',30,TEAL,True,'middle')
    f.note(1167,'这是 MoE 子层输出，之后仍需走残差路径','专家输出为人工设定；此图还没有把原输入 x 加回来。')
    f.text(48,1320,'采用 Mixtral 式选中项归一化；其他模型的路由权重规则可能不同。',24)
    out['25-moe-routing.svg']=f.save()

    f=Fig('26','专家多了，为什么不必每次全部计算？','总参数表示备有多少套能力；激活参数表示这一次实际用到了多少。',1060,1000)
    f.text(45,166,'参数账本：4 位专家，每位有 P 个参数',29,PURPLE,True)
    for i in range(4):
        x=65+237*i
        for j in [2,1,0]:f.rect(x+6*j,224+8*j,155,142,tint(PURPLE,.14),PURPLE,6)
        f.text(x+78,280,f'专家 {i+1}',27,PURPLE,True,'middle')
        f.text(x+78,330,'P',34,PURPLE,True,'middle')
    f.text(500,435,'总专家参数：4P',32,PURPLE,True,'middle')
    f.text(45,520,'当前一个 token：只调用专家 2 和 3',29,TEAL,True)
    for i in range(4):
        x=65+237*i;c=TEAL if i in (1,2) else GRAY
        f.rect(x,568,155,131,tint(c,.15),c,9)
        f.text(x+78,613,f'专家 {i+1}',27,c,True,'middle')
        if i in (1,2):mini_funnel(f,x+29,637,95,37,c)
        else:f.text(x+78,664,'本次跳过',22,c,anchor='middle')
    f.text(500,771,'本 token 激活的专家参数：2P',31,TEAL,True,'middle')
    f.note(829,'上面两笔账都还要另加共享模块与 Router','Attention 等共享计算照常执行；不是整个模型只有 2P。')
    f.lines(46,965,['通常仍需存放全部专家权重；一个 batch 也可能用到全部专家。','路由、数据搬运和负载不均都会影响速度，不能用 4÷2 推断加速倍数。'],24,step=36)
    out['26-moe-capacity.svg']=f.save()
    return out


def build():
    out={}
    f=Fig('01','一句话怎样变成另一句话？','先把源句读成一排表示，再带着目标前缀逐词生成。',1120,1060)
    f.text(50,161,'源句：I love cats',29,BLUE,True)
    f.text(595,161,'目标前缀：BOS 我',29,TEAL,True)
    for i,label in enumerate(['I','love','cats']):
        f.token(45+i*148,192,label,BLUE,w=108);f.vec(49+i*148,270,[.3,1.6,.8,2.7],BLUE,25,numeric=False)
    for i,label in enumerate(['BOS','我']):
        f.token(620+i*190,192,label,TEAL,w=108);f.vec(624+i*190,270,[.7,2.5,1.8,.4],TEAL,25,numeric=False)
    f.text(50,336,'每个 token 是一条数字带',23)
    f.text(595,336,'每个已知目标 token 也是一条数字带',22)
    for xx,ww in [(40,460),(590,430)]:
        for n in [2,1,0]:f.rect(xx+10*n,384+10*n,ww-20,300,'#f1f2ed','#c5cfc8',12)
        f.rect(xx,384,ww-20,300,'#ffffff','#a8bdb6',12)
    f.text(60,426,'Encoder 层 × N',28,BLUE,True)
    f.grid(72,466,[[.6,.2,.2],[.2,.3,.5],[.2,.4,.4]],25,TEAL,False)
    f.lines(169,487,['自注意力：源位置互相读','残差 + LN'],23)
    mini_funnel(f,86,591,102,58)
    f.lines(232,607,['FFN：各自加工','残差 + LN'],23)
    f.text(608,426,'Decoder 层 × N',28,TEAL,True)
    f.grid(617,458,[[1,None],[.4,.6]],29,TEAL,False,True)
    f.text(700,482,'只读合法目标前缀',22);f.text(700,512,'残差 + LN',21)
    f.rect(614,538,374,49,tint(ORANGE,.1))
    f.text(801,570,'交叉注意力：查询源表示',23,ORANGE,True,'middle')
    mini_funnel(f,620,617,90,42)
    f.lines(746,623,['FFN；残差 + LN','各子层单独做残差与 LN'],21,step=29)
    for x in [105,253,401]:f.line(x,298,x,378,BLUE,2,True)
    for x in [675,865]:f.line(x,298,x,378,TEAL,2,True)
    for i in range(3):f.vec(56+i*144,753,[2,.5,3,1],BLUE,28,numeric=False)
    f.line(250,685,250,741,BLUE,2,True)
    f.text(48,827,'最终源表示：生成期间保留',25,BLUE,True)
    f.path('M472,770 L546,770 L546,563 L608,563',BLUE,4,True)
    f.text(497,868,'K / V',22,BLUE)
    f.line(803,686,803,753,TEAL,2,True)
    f.vec(741,761,[1,2,2,3],TEAL,30,numeric=False)
    f.text(646,842,'末位置 → 词表打分',25)
    f.bar(638,870,220,.7,TEAL,'喜欢')
    f.bar(638,912,220,.2,GRAY,'其他')
    f.token(726,969,'喜欢',ORANGE,w=126,pending=True)
    f.path('M866,995 L1030,995 L1030,216 L942,216',ORANGE,3,True,True)
    f.text(43,939,'源句只编码一次。',25,BLUE,True)
    f.lines(43,980,['新选出的 token 接回目标输入。','选到 EOS 或满足停止条件，就结束。'],24)
    out['01-overview.svg']=f.save()

    f=Fig('02','模型为什么先把文字变成编号？','编号用来定位词表；数字大，不代表意思更重要。',790)
    f.text(420,178,'我喜欢猫',47,bold=True,anchor='middle')
    for i,(label,ident) in enumerate([('我',17),('喜欢',42),('猫',9)]):
        x=159+i*198
        f.token(x,249,label,w=126)
        f.line(x+63,300,x+63,374,BLUE,2,True)
        f.circle(x+63,422,44,tint(BLUE),BLUE,str(ident),34)
        f.text(x+63,504,'Token ID',23,color=MUTED,anchor='middle')
    f.path('M168,220 L168,207 L670,207 L670,220',GRAY)
    f.text(420,567,'一段文字 → 几块 token → 几个整数',28,bold=True,anchor='middle')
    f.note(612,'把 ID 当作“查表地址”','真实切分可能是字、子词或字节；这里按三块做示例。')
    out['02-tokenization.svg']=f.save()

    f=Fig('03','开始、结束、补齐，各用什么标记？','BOS 给起点，EOS 表示结束，PAD 只是对齐长度。',820)
    for i,t in enumerate(['BOS','我','喜欢','猫','EOS']):f.token(70+144*i,218,t,ORANGE if i==0 else RED if i==4 else BLUE,w=116)
    f.path('M82,187 L82,143 L118,155 L82,167',ORANGE,3)
    f.text(90,317,'起点',26,ORANGE,True)
    f.circle(704,162,26,tint(RED),RED,'停',24)
    f.text(630,317,'被选中后结束',25,RED,True)
    f.text(40,391,'同批长度不同，就可能补 PAD',28,bold=True)
    for row,labels in enumerate([['我','喜欢','猫'],['你好','世界','PAD']]):
        f.text(57,465+row*93,'句子 '+str(row+1),24)
        for j,t in enumerate(labels):f.token(228+j*168,432+row*93,t,GRAY if t=='PAD' else BLUE,w=131)
    f.path('M623,592 L623,636 L439,636',GRAY,2,True)
    f.text(52,678,'PAD 不是内容，要从注意力读取中排除。',27,bold=True)
    f.text(52,728,'起始 / 结束 ID 的具体规则由模型协议决定。',23,color=MUTED)
    out['03-special-tokens.svg']=f.save()

    f=Fig('04','一个编号怎样变成一串数？ ★','Embedding 是一张表；输入 ID，取出对应的一行。',820)
    f.token(46,285,'ID 17',ORANGE,w=130)
    data=[[.1,.5,-.2,.9],[.2,-.1,.7,.4],[.8,.3,.2,-.5],[.6,-.8,.1,.2]]
    for r,row in enumerate(data):
        yy=216+r*69
        if r==1:f.rect(266,yy-7,435,66,tint(ORANGE,.08),ORANGE)
        f.text(306,yy+34,str(16+r),25,anchor='middle')
        f.vec(363,yy,row,ORANGE if r==1 else BLUE,69)
    f.text(301,177,'ID',25,bold=True,anchor='middle');f.text(501,177,'嵌入表 E 的特征列',27,bold=True,anchor='middle')
    f.line(181,310,260,310,ORANGE,4,True)
    f.text(80,547,'取出第 17 行',29,ORANGE,True)
    f.path('M702,310 L746,310 L746,577 L596,577',ORANGE,3,True)
    f.vec(317,549,[.2,-.1,.7,.4],ORANGE,70)
    f.note(666,'现在有了可计算的向量，但还没有句内上下文','同一个 ID 先查到同一行；后面再结合位置与上下文。')
    out['04-embedding.svg']=f.save()

    f=Fig('05','同一个词，换个位置为什么不一样？ ★','内容相同，再加入不同的位置表示，后续输入就不同。',1000)
    f.text(40,164,'经典加法位置编码：两个位置上的同一个 token',26,bold=True)
    for r,(pos,pe,res) in enumerate([(0,[0,1],[2,2]),(1,[.84,.54],[2.84,1.54])]):
        y=234+r*144
        f.text(39,y+29,f'位置 {pos}',26,bold=True)
        f.vec(183,y,[2,1],BLUE,57);f.text(316,y+34,'+',32)
        f.vec(356,y,pe,ORANGE,57);f.text(490,y+34,'=',32)
        f.vec(532,y,res,TEAL,76)
    for x,t,c in [(183,'内容',BLUE),(356,'位置',ORANGE),(532,'相加后的表示',TEAL)]:f.text(x,204,t,25,c,True)
    f.lines(42,542,['同一份内容 + 不同站位 → 不同表示。','例：二维正弦 PE；内容列已包含 Embedding 的缩放。'],24)
    f.line(32,606,808,606,'#d7ddd9')
    f.text(40,653,'另一种方案 RoPE：转动 Q / K 的方向',28,PURPLE,True)
    cx,cy,r=205,794,98
    f.circle(cx,cy,r,BG,'#d9d6e0');f.line(cx-r-15,cy,cx+r+22,cy,GRAY,1,True);f.line(cx,cy+r+8,cx,cy-r-19,GRAY,1,True)
    f.line(cx,cy,cx+84.9,cy-49,BLUE,4,True)
    f.line(cx,cy,cx,cy-98,PURPLE,4,True)
    f.path(f'M{cx+43},{cy-25} A50,50 0 0 0 {cx},{cy-50}',ORANGE,3,True)
    f.text(298,747,'原方向',22,BLUE)
    f.lines(395,733,['像转动指针，长度保持。','不同位置转不同角度，','匹配分数能反映相对距离。'],25)
    f.text(395,872,'这是另一条位置编码路线，',23,color=MUTED);f.text(395,907,'不是在上半图之后再做一次。',23,color=MUTED)
    out['05-position.svg']=f.save()

    f=Fig('06','Q、K、V 为什么要分三份？ ★','同一份表示，分别改写成“怎么问、怎么匹配、拿回什么”。',870)
    f.text(40,162,'同一个输入向量 x = [1, 2]',29,bold=True)
    for i,(label,desc,w,result,color) in enumerate([
        ('Q','查询条件',[[1,0],[0,1]],[1,2],PURPLE),
        ('K','匹配索引',[[1,1],[0,1]],[1,3],ORANGE),
        ('V','读取内容',[[1,0],[1,2]],[3,4],TEAL)]):
        y=233+i*169
        f.circle(66,y+48,29,tint(color),color,label,28)
        f.text(115,y+11,desc,24,color,True)
        f.vec(115,y+30,[1,2],BLUE,47);f.text(231,y+61,'×',33)
        f.grid(279,y,w,46,color,True);f.text(389,y+61,'=',33)
        f.vec(436,y+30,result,color,59)
        f.text(598,y+62,'用法不同',24,color)
    f.note(735,'Q 和 K 决定“读多少”，V 决定“读到什么”','数字只是手工投影例子；三份向量不必相同。')
    out['06-qkv.svg']=f.save()

    f=Fig('07','注意力怎样把信息按比例拿回来？ ★','先分配读取权重，再把各个 Value 按权重相加。',1090)
    names=['位置 1','位置 2','位置 3']
    for i,t in enumerate(names):f.token(167+i*196,162,t,TEAL if i<2 else GRAY,w=124)
    f.text(38,273,'分数',25,bold=True)
    for i,t in enumerate(['ln 2','0','7']):f.text(228+i*196,274,t,30,anchor='middle')
    f.text(38,333,'掩码',25,bold=True)
    for i,t in enumerate(['保留','保留','禁止']):f.text(228+i*196,334,t,27,RED if i==2 else TEAL,anchor='middle')
    f.path('M590,247 L648,280 M648,247 L590,280',RED,3)
    f.text(623,391,'−∞',29,RED,True,'middle')
    f.text(39,397,'取指数，再把总和变成 1',22,color=MUTED)
    f.text(39,448,'权重',25,bold=True)
    for i,(v,t) in enumerate([(2/3,'2/3'),(1/3,'1/3'),(0,'0')]):
        f.rect(167+i*196,507-130*v,122,130*v,tint(TEAL,.55) if v else tint(GRAY),radius=3)
        f.text(228+i*196,547,t,30,TEAL if v else GRAY,True,'middle')
    f.text(170,597,'每一份权重，对应下面的一份内容',24,color=MUTED)
    for i,(v,mult) in enumerate([([3,0],'× 2/3'),([0,6],'× 1/3'),([100,100],'× 0')]):
        f.vec(164+i*196,637,v,TEAL if i<2 else GRAY,65)
        f.text(228+i*196,748,mult,27,anchor='middle')
    f.vec(164,792,[2,0],TEAL,65);f.text(319,832,'+',32)
    f.vec(360,792,[0,2],TEAL,65);f.text(515,832,'+',32)
    f.vec(556,792,[0,0],GRAY,65)
    f.text(230,930,'汇总结果',29,bold=True);f.text(401,931,'=',32);f.vec(450,891,[2,2],PURPLE,70)
    f.text(40,1024,'被屏蔽的位置即使分数很高，也贡献 0。',27,RED,True)
    out['07-scaled-attention.svg']=f.save()

    f=Fig('08','多个头，多出了什么能力？ ★','对同一批位置保留几种不同读法，再合并这些读法的结果。',960)
    for col,(color,title,weights) in enumerate([(BLUE,'头 1',[.7,.2,.1]),(PURPLE,'头 2',[.1,.2,.7])]):
        x=46+col*404
        f.text(x+158,173,title,31,color,True,'middle')
        for j,t in enumerate(['我','喜欢','猫']):
            f.token(x+j*112,211,t,color,w=91)
            f.line(x+j*112+45,263,x+158,363,color,1+weights[j]*14,True,opacity=.7)
        f.text(x+158,406,'同一个 Query',24,color,anchor='middle')
        for j,v in enumerate(weights):f.bar(x+30,443+j*42,200,v,color,f'{v:.1f}',h=19)
        f.text(x+158,600,'各自得到 2 维结果',23,color,anchor='middle')
        f.vec(x+99,628,['a','b'] if col==0 else ['c','d'],color,60)
    f.line(265,691,343,758,BLUE,3,True);f.line(669,691,505,758,PURPLE,3,True)
    f.vec(303,774,['a','b'],BLUE,59);f.vec(421,774,['c','d'],PURPLE,59)
    f.text(45,812,'拼接 2 + 2 = 4 维',24,bold=True)
    f.text(562,812,'再由 W_O 混合',24,bold=True)
    f.text(42,901,'各头都读完整的合法位置；不是每头分一段句子。',26,TEAL,True)
    out['08-multi-head.svg']=f.save()

    f=Fig('09','补齐的空位，为什么不能一起读？ ★','Padding mask 排除占位符，否则它仍会分走权重。',900)
    for j,t in enumerate(['我','喜欢','猫','PAD']):
        f.token(215+j*143,156,t,GRAY if j==3 else BLUE,w=95)
    f.text(38,282,'假设原始分数都为 0',28,bold=True)
    for row,(title,weights) in enumerate([('没屏蔽',[.25]*4),('屏蔽 PAD',[1/3,1/3,1/3,0])]):
        y=350+row*170
        f.text(39,y+19,title,25,RED if row==0 else TEAL,True)
        for j,v in enumerate(weights):
            xx=215+j*143
            f.rect(xx,y+60-v*135,95,v*135,GRAY if j==3 else BLUE,radius=2)
            f.text(xx+47,y+105,'1/4' if row==0 else ('1/3' if j<3 else '0'),27,RED if j==3 else INK,anchor='middle')
        f.line(205,y+61,779,y+61,GRAY,1)
    f.note(707,'PAD 变成零向量，也不能代替 mask','分数为 0 仍有 exp(0)=1；屏蔽后才不参与分配。')
    out['09-padding-mask.svg']=f.save()

    f=Fig('10','残差连接：保留底稿，再叠加修订 ★','输出 = 原表示 + 子层给出的变化；不是把两份向量拼起来。',900)
    ox,oy,scale=160,640,125
    f.line(ox-25,oy,ox+3.6*scale,oy,GRAY,2,True);f.line(ox,oy+25,ox,oy-3.7*scale,GRAY,2,True)
    for i in range(1,4):
        f.line(ox+i*scale,oy-4,ox+i*scale,oy+4,GRAY);f.text(ox+i*scale,oy+34,str(i),21,anchor='middle')
        f.line(ox-4,oy-i*scale,ox+4,oy-i*scale,GRAY);f.text(ox-23,oy-i*scale+7,str(i),21,anchor='middle')
    a=(ox+2*scale,oy-scale);b=(ox+scale,oy-3*scale)
    f.line(ox,oy,*a,BLUE,7,True)
    f.line(*a,*b,ORANGE,7,True)
    f.line(ox,oy,*b,TEAL,5,True)
    f.text(430,558,'底稿 x = [2, 1]',27,BLUE,True)
    f.lines(427,338,['修订 Δ = [−1, 2]','从蓝箭头的终点出发'],25,ORANGE)
    f.lines(45,182,['合起来 y = [1, 3]','绿箭头直接指向最终结果'],26,TEAL)
    f.text(420,744,'[2, 1] + [−1, 2] = [1, 3]',34,bold=True,anchor='middle')
    f.text(47,826,'每个分量对应相加，所以两边必须同宽。',27)
    out['10-residual.svg']=f.save()

    f=Fig('11','LayerNorm：把每个 token 的刻度整理好 ★','分别处理每个向量，不把整句话的数字混在一起求平均。',950)
    for row,(title,vals,mean,sigma,color) in enumerate([('token A',[1,3],2,1,BLUE),('token B',[10,14],12,2,PURPLE)]):
        y=226+row*185
        f.text(40,y-38,f'{title}  {vals}',28,color,True)
        f.line(90,y,738,y,GRAY,2)
        for j in range(0,16,3):
            xx=90+j*43.2;f.line(xx,y-5,xx,y+5,GRAY);f.text(xx,y+36,str(j),21,anchor='middle')
        for v in vals:f.circle(90+v*43.2,y,11,color,'none')
        f.line(90+mean*43.2,y-26,90+mean*43.2,y+12,color,2,dash=True)
        f.text(740,y-34,f'μ={mean}',24,color,anchor='end')
        f.text(40,y+83,f'各自减去均值 {mean}，再除以标准差 {sigma}',24,color)
    f.text(40,587,'整理后（暂设 γ=1、β=0，忽略 ε）',26,bold=True)
    for row,color in enumerate([BLUE,PURPLE]):
        y=648+row*73
        f.line(233,y,627,y,GRAY,2)
        for v in [-1,0,1]:
            xx=430+v*135
            if v:f.circle(xx,y,11,color,'none')
            else:f.line(xx,y-12,xx,y+12,GRAY)
        f.text(165,y+8,'A' if row==0 else 'B',26,color,True)
    for x,t in [(295,'−1'),(430,'0'),(565,'1')]:f.text(x,771,t,25,anchor='middle')
    f.text(40,835,'相对分布被整理到同一尺度；再由 γ、β 调整刻度。',25)
    f.text(40,884,'它没有把两个 token 变成同一种含义。',24,color=MUTED)
    out['11-layernorm.svg']=f.save()

    f=Fig('12','归一化放前面、放后面，有什么不同？ ★','顺着主线看位置，再看旁路：残差始终要认清加回谁。',780)
    for col,(title,color,ys) in enumerate([('Post-LN',BLUE,[('F',337),('+',471),('LN',605)]),('Pre-LN',PURPLE,[('LN',337),('F',471),('+',605)])]):
        x=223+col*397
        f.text(x,173,title,31,color,True,'middle')
        f.vec(x-81,210,[1,2,3],color,55)
        f.text(x+109,245,'x',27,color)
        for j,(label,y) in enumerate(ys):
            f.line(x,263 if j==0 else ys[j-1][1]+32,x,y-34,color,3,True)
            f.circle(x,y,33,tint(color),color,label,25)
        dest=471 if col==0 else 605
        f.path(f'M{x-85},236 L{x-139},236 L{x-139},{dest} L{x-37},{dest}',ORANGE,4,True)
        f.text(x-144,405,'保留 x',22,ORANGE,anchor='end')
        f.text(x,695,'LN(x + F(x))' if col==0 else 'x + F(LN(x))',27,color,True,'middle')
    out['12-norm-order.svg']=f.save()

    f=Fig('13','FFN：每个 token 各自加工，参数共用 ★','Attention 在位置之间读信息；FFN 在单个位置内部组合特征。',960)
    f.text(216,171,'输入 d',27,BLUE,True,'middle');f.text(440,171,'中间 d_ff',27,PURPLE,True,'middle');f.text(664,171,'输出 d',27,TEAL,True,'middle')
    for row,label in enumerate(['我','喜欢','猫']):
        y=258+row*184
        f.token(35,y+11,label,w=105)
        inys=[y+18,y+67];midys=[y-18+j*23 for j in range(6)];outys=[y+18,y+67]
        for xx,aa,xx2,bb in [(216,inys,440,midys),(440,midys,664,outys)]:
            for ya in aa:
                for yb in bb:f.line(xx,ya,xx2,yb,PURPLE,1.4,opacity=.25)
        for xx,yy,c in [(216,inys,BLUE),(440,midys,PURPLE),(664,outys,TEAL)]:
            for y1 in yy:f.circle(xx,y1,9,c,'none')
        f.line(33,y+131,779,y+131,'#e2e4df',1,dash=True)
    f.text(40,835,'同一套权重，分别用于三个 token；没有跨行连线。',25,bold=True)
    f.text(40,886,'每行内部先展开，再非线性加工，最后回到原宽度。',25)
    out['13-ffn.svg']=f.save()

    f=Fig('14','一个 Encoder 层怎样更新三个位置？','先互相读取，再分别加工；每步保留 token 数。',1060)
    xs=[205,423,641]
    for x,t in zip(xs,['I','love','cats']):f.token(x-56,155,t,w=112);f.vec(x-60,232,[1,2,3],BLUE,41,numeric=False)
    f.text(36,314,'自注意力：各位置可以互相读',27,TEAL,True)
    for a,x in enumerate(xs):
        for b,z in enumerate(xs):f.line(x,345,z,453,TEAL,3 if a==b else 1.6,True,opacity=.5)
    for x in xs:
        f.circle(x,487,19,tint(ORANGE),ORANGE,'+',23)
        f.path(f'M{x+65},250 L{x+79},250 L{x+79},487 L{x+21},487',ORANGE,2,True)
        f.text(x,541,'LN',24,anchor='middle');f.vec(x-60,573,[3,1,2],TEAL,41,numeric=False)
    f.text(35,668,'FFN：位置之间不再连线',27,PURPLE,True)
    for x in xs:
        mini_funnel(f,x-60,712,120,80)
        f.circle(x,850,19,tint(ORANGE),ORANGE,'+',23)
        f.path(f'M{x+66},594 L{x+85},594 L{x+85},850 L{x+21},850',ORANGE,2,True)
        f.text(x,898,'LN',24,anchor='middle');f.vec(x-60,924,[2,3,1],TEAL,41,numeric=False)
    f.text(40,1000,'堆叠 N 层后，仍是一排源位置表示，供 Decoder 查询。',25)
    out['14-encoder.svg']=f.save()

    f=Fig('15','“我”能看见谁，不能看见谁？ ★','当前输入位置预测下一个 token；允许看自己，屏蔽后面。',950)
    labels=['BOS','我','喜欢','猫']
    f.text(335,166,'被读取的位置（Key）',27,bold=True,anchor='middle')
    for j,t in enumerate(labels):f.text(267+j*100,217,t,27,anchor='middle')
    f.text(38,320,'查询',25,bold=True);f.text(38,357,'位置',25,bold=True)
    for i,t in enumerate(labels):f.text(170,302+i*100,t,26,anchor='end')
    data=[[.16 if j<=i else None for j in range(4)] for i in range(4)]
    f.grid(218,241,data,100,TEAL,False,True)
    for i in range(4):
        for j in range(4):f.text(267+j*100,302+i*100,'可读' if j<=i else '遮住',24,TEAL if j<=i else RED,anchor='middle')
    f.rect(209,333,419,111,'none',ORANGE,6)
    f.text(653,382,'看这一行',24,ORANGE,True)
    f.text(45,712,'输入到“我”时：',28,bold=True)
    for j,t in enumerate(labels):f.token(45+j*154,753,t,TEAL if j<2 else GRAY,w=124,pending=j>=2)
    f.line(700,778,756,778,ORANGE,3,True)
    f.text(420,873,'读取 BOS 和“我”，然后预测“喜欢”。',28,ORANGE,True,'middle')
    out['15-causal-mask.svg']=f.save()

    f=Fig('16','交叉注意力：写到这里，该查源句哪里？ ★','左边是已读懂的源句，右边是正在形成的目标表示。',960)
    f.text(47,168,'Encoder 源表示 → K / V',27,BLUE,True)
    f.text(511,168,'Decoder 状态 → Q',27,PURPLE,True)
    for i,t in enumerate(['I','love','cats']):
        y=225+i*131;f.token(73,y,t,BLUE,w=127);f.vec(74,y+63,[1,3,2],BLUE,42,numeric=False)
    for i,t in enumerate(['BOS','我']):
        y=263+i*186;f.token(610,y,t,PURPLE,w=125)
    weights=[[.4,.4,.2],[.1,.2,.7]]
    for i,row in enumerate(weights):
        for j,v in enumerate(row):f.path(f'M603,{288+i*186} C440,{288+i*186} 393,{251+j*131} 210,{251+j*131}',TEAL,2+v*13,True,opacity=.57)
    f.text(440,591,'线越粗，读取比例越大',24,TEAL,True,'middle')
    f.text(58,669,'同一关系画成表：',26,bold=True)
    for j,t in enumerate(['I','love','cats']):f.text(347+j*72,696,t,25,anchor='middle')
    f.text(245,754,'BOS',25,anchor='end');f.text(245,825,'我',25,anchor='end')
    f.grid(311,715,weights,72,TEAL)
    f.text(613,768,'2 个目标位置',24);f.text(613,808,'× 3 个源位置',24)
    f.text(42,896,'源句早已给定，源位置靠后不等于目标侧的“未来”。',25)
    out['16-cross-attention.svg']=f.save()

    f=Fig('17','Decoder 的三道工序，各处理什么？','先整理已写内容，再查询源句，最后加工融合后的特征。',1080)
    xs=[303,546]
    for x,t in zip(xs,['BOS','我']):f.token(x-57,155,t,TEAL,w=114)
    f.text(40,260,'① 目标内部：只读合法前缀',28,TEAL,True)
    for x in xs:f.line(x,304,x,371,TEAL,4,True)
    f.line(xs[0],304,xs[1],371,TEAL,4,True)
    f.line(xs[1],304,xs[0],371,RED,2,dash=True)
    f.text(676,341,'禁止逆向',22,RED)
    for x in xs:f.vec(x-59,397,[1,3,2],TEAL,41,numeric=False)
    f.text(420,477,'+ 残差 · LN',23,ORANGE,anchor='middle')
    f.text(40,552,'② 目标查询源句：交叉注意力',28,BLUE,True)
    for i,t in enumerate(['I','love','cats']):
        f.token(40,590+i*60,t,BLUE,w=100)
        for x in xs:f.line(154,615+i*60,x-70,668,BLUE,2,True,opacity=.5)
    for x in xs:f.vec(x-59,646,[3,2,1],BLUE,41,numeric=False)
    f.text(420,757,'+ 残差 · LN',23,ORANGE,anchor='middle')
    f.text(40,826,'③ 各自加工：FFN',28,PURPLE,True)
    for x in xs:mini_funnel(f,x-60,865,120,67)
    f.text(420,984,'+ 残差 · LN → 下一 Decoder 层',26,ORANGE,anchor='middle')
    out['17-decoder.svg']=f.save()

    f=Fig('18','一串数字怎样变成下一个 token？ ★','LM Head 给词表候选打分；解码策略再做选择。',960)
    f.text(35,171,'最后有效位置的表示',27,bold=True)
    f.vec(416,142,[1,2,3,1],TEAL,54)
    f.path('M701,171 L750,171 L750,280 L662,280',TEAL,3,True)
    f.text(36,241,'词表分数 logits',27,BLUE,True)
    logits=[math.log(6),math.log(3),0]
    for i,(label,value) in enumerate(zip(['我','你','EOS'],logits)):
        y=289+i*64;f.text(91,y+24,label,25,anchor='middle');f.bar(167,y,455,value/2,BLUE,f'{value:.2f}')
    f.text(36,521,'Softmax：转成概率',27,PURPLE,True)
    for i,(label,value) in enumerate(zip(['我','你','EOS'],[.6,.3,.1])):
        y=568+i*64;f.text(91,y+24,label,25,anchor='middle');f.bar(167,y,455,value,PURPLE,f'{value:.0%}')
    f.note(788,'Greedy 选最大项“我”；采样则按概率抽取','EOS 分数是 0，概率仍是 10%；0 分不等于被屏蔽。')
    out['18-output-head.svg']=f.save()

    f=Fig('19','每轮究竟把哪个 token 送回模型？ ★','已处理的前缀留在左边，刚选出的下一个 token 放在右边。',940)
    labels=['BOS','我','喜欢','猫','EOS']
    for row in range(4):
        y=189+row*155
        f.text(40,y-19,f'第 {row+1} 轮',26,bold=True)
        for j in range(row+1):
            f.token(48+j*133,y,labels[j],ORANGE if j==row else BLUE,w=109)
        nx=48+(row+1)*133
        f.line(nx-16,y+26,nx+3,y+26,GRAY,2,True)
        f.token(nx+11,y,labels[row+1],RED if row==3 else TEAL,w=109,pending=True)
        f.text(57,y+97,'本轮已处理',23,BLUE)
        f.text(nx+16,y+97,'刚选出' if row<3 else '停止',23,RED if row==3 else TEAL)
    f.text(39,848,'实框：已有表示 / KV；虚框：还没作为输入计算。',25,bold=True)
    out['19-generation-loop.svg']=f.save()

    f=Fig('20','升维、非线性、降维，各干了什么？ ★','把两个数展开成更多组合，分别处理，再汇总成两个数。',1160,960)
    cols=[(104,'输入 2 维',BLUE),(324,'上投影 4 维',PURPLE),(574,'ReLU 后',ORANGE),(831,'下投影 2 维',TEAL)]
    for x,label,c in cols:f.text(x,169,label,27,c,True,'middle')
    ys2=[325,475];ys4=[253,351,449,547]
    # x=[a,b] -> [a,b,a+b,a-b], zero weights omitted.
    edges=[(0,0,1),(1,1,1),(0,2,1),(1,2,1),(0,3,1),(1,3,-1)]
    for i,j,sign in edges:f.line(136,ys2[i],288,ys4[j],BLUE if sign>0 else RED,2.2,True,sign<0,opacity=.65)
    for j in range(4):f.line(358,ys4[j],539,ys4[j],GRAY,2,True)
    for j,i,sign in [(0,0,1),(2,0,1),(1,1,1),(3,1,-1)]:f.line(611,ys4[j],795,ys2[i],TEAL if sign>0 else RED,2.4,True,sign<0,opacity=.7)
    for x,ys,vals,c in [(104,ys2,[-1,2],BLUE),(324,ys4,[-1,2,1,-3],PURPLE),(574,ys4,[0,2,1,0],ORANGE),(831,ys2,[1,2],TEAL)]:
        for y,v in zip(ys,vals):f.circle(x,y,32,tint(c,.18) if v else '#f0efeb',c if v else GRAY,v,28)
    for j,t in enumerate(['a','b','a+b','a−b']):f.text(324,ys4[j]-44,t,24,PURPLE,anchor='middle')
    f.text(452,304,'负值 → 0',23,RED,True,'middle');f.text(452,522,'负值 → 0',23,RED,True,'middle')
    f.text(833,611,'r₁+r₃，r₂−r₄',23,TEAL,anchor='middle')
    f.text(40,658,'线是加权组合，不是复制：虚线表示本例的负权重。',24,color=MUTED)
    f.line(32,694,928,694,'#d7ddd9')
    f.text(39,744,'为什么中间需要非线性？',30,bold=True)
    ox,oy=198,936
    f.line(78,oy,334,oy,GRAY,1.6,True);f.line(ox,1000,ox,805,GRAY,1.6,True)
    f.path(f'M78,{oy} L{ox},{oy} L315,819',ORANGE,5)
    f.text(133,805,'ReLU',26,ORANGE,True);f.text(254,985,'负数归零',22,RED,anchor='middle')
    f.lines(396,815,['更多组合，各自有不同响应。','负数被压到零，正数保留。','下投影再利用这些加工结果。'],26)
    f.lines(40,1061,['若没有非线性，两次线性投影就能合成一次。','升维本身不会增加输入事实；降维也不是复原原输入。'],25,color=TEAL,step=38)
    out['20-up-down-projection.svg']=f.save()

    f=Fig('21','KV Cache：留住旧笔记，只补新的一页 ★','每层保留自己的 K / V；新 Query 仍要读取全部合法历史。',1070)
    f.text(44,171,'第 1 步：prompt 已处理，刚选出 y₁',28,bold=True)
    tokens=['p₁','p₂','p₃']
    for row in range(3):
        y=222+row*91
        f.text(45,y+34,f'层 {row+1}',25,bold=True)
        for j,t in enumerate(tokens):
            f.rect(159+j*91,y,78,59,tint(BLUE,.2));f.text(198+j*91,y+24,t,21,BLUE,anchor='middle');f.text(198+j*91,y+49,'K / V',20,BLUE,anchor='middle')
    f.token(548,277,'y₁',ORANGE,w=108,pending=True)
    f.lines(534,391,['刚选出，还没算它的 KV'],23,ORANGE)
    f.line(32,503,808,503,'#d7ddd9')
    f.text(44,558,'第 2 步：输入 y₁，逐层算出并追加它的 KV',27,bold=True)
    for row in range(3):
        y=610+row*92
        f.text(43,y+34,f'层 {row+1}',25,bold=True)
        for j,t in enumerate(tokens+['y₁']):
            col=BLUE if j<3 else ORANGE
            f.rect(159+j*91,y,78,59,tint(col,.2));f.text(198+j*91,y+24,t,21,col,anchor='middle');f.text(198+j*91,y+49,'K / V',20,col,anchor='middle')
        f.circle(641,y+28,29,tint(TEAL),TEAL,'q',27)
        f.path(f'M159,{y+65} L159,{y+76} L511,{y+76} L511,{y+65}',TEAL,2)
        f.path(f'M607,{y+28} L557,{y+28} L557,{y+76} L516,{y+76}',TEAL,3,True)
    f.text(625,920,'y₁ 的 Query',23,TEAL,anchor='middle')
    f.text(39,992,'全部层完成后选出 y₂；此刻仍没有 y₂ 的 KV。',27,bold=True)
    out['21-kv-cache.svg']=f.save()

    f=Fig('22','Decoder-only：在同一条序列后继续写 ★','prompt 和已生成 token 走同一套层，没有独立的源句编码器。',1130)
    for i,t in enumerate(['我','喜欢','猫']):f.token(181+i*170,156,t,w=112);f.vec(181+i*170,234,[1,3,2],BLUE,38,numeric=False)
    f.text(40,318,'一排 token 表示',25,bold=True)
    for i in [2,1,0]:f.rect(98+10*i,358+10*i,632,420,'#f0f1eb','#c3cec5',11)
    f.rect(98,358,632,420,'white','#b0c1b5',11)
    f.text(121,405,'一层：Pre-RMSNorm + RoPE + 门控 FFN',25,bold=True)
    f.grid(138,445,[[1,None,None],[.4,.6,None],[.2,.3,.5]],53,TEAL,False,True)
    f.lines(370,472,['RMSNorm → 因果注意力','Q / K 应用 RoPE','读取与追加本层 KV','再加回原输入'],25,step=37)
    for x in [147,342,537]:mini_funnel(f,x,660,118,57)
    f.text(414,753,'RMSNorm → 门控 FFN → 残差',25,PURPLE,anchor='middle')
    f.text(720,838,'× N 层',26,anchor='end',bold=True)
    f.line(417,790,417,867,BLUE,3,True)
    f.text(418,907,'最终 RMSNorm → LM Head',28,bold=True,anchor='middle')
    f.bar(224,945,330,.7,TEAL,'候选 A')
    f.bar(224,986,330,.3,PURPLE,'候选 B')
    f.path('M710,965 L797,965 L797,182 L709,182',ORANGE,3,True,True)
    f.text(42,1064,'选出的新 token 接回序列；满足停止条件就结束。',25)
    out['22-decoder-only.svg']=f.save()
    out.update(architecture_additions())
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    diagrams=build();bad=[]
    OUT.mkdir(parents=True,exist_ok=True)
    for name,content in diagrams.items():
        p=OUT/name
        if args.check:
            if not p.exists() or p.read_text()!=content:bad.append(name)
        else:p.write_text(content,encoding='utf-8')
    if bad:raise SystemExit('Diagram drift: '+', '.join(bad))
    print(f'{"Checked" if args.check else "Built"} {len(diagrams)} example-based Transformer figures')


if __name__=='__main__':main()
