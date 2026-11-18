"""Display equations for the foundational lessons and their worked examples."""

from html import escape

from transformer_equations import EQUATIONS, call, equals, frac, mi, mn, mo, row, sqrt, sub, tag, vector


def text(value):
    return tag('mtext', escape(value))


def ratio(name, numerator, denominator, *tail):
    return equals(text(name), frac(numerator, denominator), *tail)


def matrix(values):
    rows = (tag('mtr', *(tag('mtd', mn(value)) for value in items)) for items in values)
    return row(mo('['), tag('mtable', *rows), mo(']'))


n = mi('N')
t0, t1, tn = sub('t', 0), sub('t', 1), sub('t', 'N')

FOUNDATION_EQUATIONS = {
    'lookup': (equals(row(vector(0, 1, 0, 0), mi('E')), row(mi('E'), mo('['), mn(1), mo(']')),
                      mo('='), vector('0.2', '0.8', '−0.1')),),
    'conditional': (equals(call('P', sub('y', 1), mo(','), sub('y', 2), mo('∣'), mi('x')),
                           call('P', sub('y', 1), mo('∣'), mi('x')), mo('·'),
                           call('P', sub('y', 2), mo('∣'), mi('x'), mo(','), sub('y', 1))),),
    'greedy': EQUATIONS['greedy'],
    'sampling': EQUATIONS['sampling'],
    'attention_scores': (
        equals(sub('s', 'ij'), frac(row(sub('q', 'i'), mo('·'), sub('k', 'j')), sqrt(sub('d', 'k'))),
               mo('+'), sub('M', 'ij')),
        *EQUATIONS['weights'],
    ),
    'attention_example': (
        row(mn('0.75'), mo('×'), mn(2), mo('+'), mn('0.25'), mo('×'), mn(8), mo('='), mn('3.5')),
        row(mn('0.25'), mo('×'), mn(2), mo('+'), mn('0.75'), mo('×'), mn(8), mo('='), mn('6.5')),
    ),
    'cache_work': (
        equals(text('逐轮重算'), mn(4), mo('+'), mn(5), mo('+'), mn(6), mo('='), mn(15)),
        equals(text('复用历史'), mn(4), mo('+'), mn(1), mo('+'), mn(1), mo('='), mn(6)),
    ),
    'cache_bytes': (equals(sub('B', 'KV'), mn(2), mo('·'), mi('L'), mo('·'), mi('T'),
                           mo('·'), sub('H', 'KV'), mo('·'), mi('d'), mo('·'), mi('b')),),
    'queue_wait': (equals(sub('W', 'i'), sub('t', '发出'), mo('−'), sub('t', '到达')), ),
    'matmul': (equals(row(matrix(((1, 2), (3, 4))), matrix(((2,), (3,)))), matrix(((8,), (18,)))),),
    'resource_bound': (
        ratio('搬运时间下界', text('字节数'), text('带宽')),
        ratio('计算时间下界', text('运算量'), text('算力')),
        row(sub('T', '总'), mo('≥'), call('max', sub('T', '搬运'), mo(','), sub('T', '计算'))),
    ),
    'arithmetic_intensity': (ratio('算术强度', text('浮点运算量'), text('搬运字节数')), ),
    'parallel_merge': (equals(mi('o'), frac(row(sub('Z', 0), sub('o', 0), mo('+'), sub('Z', 1), sub('o', 1)),
                                          row(sub('Z', 0), mo('+'), sub('Z', 1)))),),
    'reduce_sum': (equals(row(vector(1, 2, 3), mo('+'), vector(10, 20, 30), mo('+'),
                              vector(100, 200, 300)), vector(111, 222, 333)),),
    'transfer_bound': (row(sub('T', '搬运'), mo('≥'), frac(sub('B', 'KV'), sub('R', '带宽'))),),
    'spec_accept': (equals(call('α', mi('x')), call('min', mn(1), mo(','),
                              frac(call('p', mi('x')), call('q', mi('x'))))),),
    'spec_reject': (
        equals(call('r', mi('x')), call('max', mn(0), mo(','), call('p', mi('x')), mo('−'), call('q', mi('x')))),
        equals(call('p′', mi('x')), frac(call('r', mi('x')), row(tag('munder', mo('∑'), mi('v')), call('r', mi('v'))))),
    ),
    'metric_intervals': (
        equals(mi('TTFT'), t1, mo('−'), t0),
        equals(sub('ITL', 'i'), sub('t', 'i'), mo('−'), sub('t', 'i−1')),
        ratio('TPOT', row(tn, mo('−'), t1), row(n, mo('−'), mn(1))),
        equals(sub('E2E', '末项'), tn, mo('−'), t0),
    ),
    'nearest_rank': (equals(mi('k'), mo('⌈'), mn('0.95'), mo('·'), mi('n'), mo('⌉')),),
    'throughput': (
        ratio('请求吞吐', text('窗口内完成请求数'), text('窗口时长')),
        ratio('输出吞吐', text('窗口内计入的输出 token 数'), text('窗口时长')),
    ),
}


def foundation_math(names):
    formulas = ''.join(f'<math display="block">{formula}</math>'
                       for name in names for formula in FOUNDATION_EQUATIONS[name])
    return f'<div class="foundation-math">{formulas}</div>' if formulas else ''
