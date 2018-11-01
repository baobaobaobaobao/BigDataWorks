# -*- coding: utf-8 -*-
import sys
import jieba


def print_dict(pdict, filename):
    # reload(sys)
    # sys.setdefaultencoding("utf-8")
    fp = open(filename, "w")

    for k, v in pdict.items():
        if k == '':
            continue
        fp.write(k)
        fp.write(":")
        fp.write(str(v))
        fp.write("\n")
    fp.close()
    print
    ('write to file over!')


def print_words(text, filename):
    split_text = text.split('/')
    out_list = []
    out_dict = {}

    for it in split_text:
        if it in out_list:
            out_dict[it] = 1 + out_dict[it]
        else:
            out_list.append(it)
            out_dict[it] = 1

    print_dict(out_dict, filename)


def jieba_cut(cont):
    seg_list = jieba.cut(cont, cut_all=True)
    seg_str = "/".join(seg_list)
    print
    ( 'cut is OK!')
    return seg_str


def main():
    cont = "周三大盘早盘出现单边下挫。直接跌破前期2030点的中继平台，向2000点附近寻求支撑，空头来势汹汹，恐慌情绪开始蔓延。其实可能引发大盘杀跌的原因归结起来也就那么几个，六月钱荒预期、世界杯魔咒，新股联网测试即将发行。但这些因素都是市场可以预见的，可以说是压力，但是算不上重大利空。不过是主力资金借机发挥的借口而已。在没有重大利空出现的背景下，昨日的单边杀跌后缓慢回升的走势更像是主力刻意砸盘，完成低吸之后的回撤。因为虽然短线破位杀跌，但成交量并未放大，依然萎缩至600亿一下的地量水平，说明并没有出现资金的大规模出逃。而且盘口这么轻，主力想要玩玩指数游戏并不难。从盘面上看，上证指数尚未跌至2000点，盘中就开始企稳回升，日K线的长下影线就是主力完成低吸之后回撤所导致的被动式拉升。可见杀跌不是目的，洗盘吸筹才是主力的真实意图。从这个意义上说，市场内在的做多动能依然还在，只是暂时被压抑着，只要获得喘息的机会，反弹便有可能卷土重来。从期指上看，短线投机资金入场的痕迹较明显。从前20名席位的增减仓的力度来看，多方相对弱势，空方参与的积极程度更高。减持多单较多的上海东证期货席位，减持多单534手。相比前日增持403手的多单，前后相差不远，短线投机资金的属性非常明显。空方减持的席位中，减持空单最多的光大期货席位，减持609手空单。和前日增持927手空单对照来看，显然是短空资金获利离场。总的来看，昨日的单边下挫主要是短线投机资金所为，随着这部分获利空单的兑现离场，空头的势头将大大减弱，大盘将获得休整的机会。综上所述，笔者认为昨日的杀跌不过是资金炒作的短期行为，绝对不会持续太久。短期的技术性反抽随即就会展开。主力的洗盘吸筹动作恰恰是反弹来临的前兆。与此同时，新股上市在即，大盘连续地价地量也使得政策加码更加值得期待。多年经验表明，无论市场强势还是弱势，真正值得跟随的，当属那些走势独立于大盘、题材想象空间大、机构运作底气充足的主力控盘股。由于受主力关照，此类个股进可攻、退可守，投资者不妨集中精力挖掘跟随，波段操作。详情点击：目前这一批个股主力已完成建仓，即将拉升。 "
    seg_str = jieba_cut(cont)
    print_words(seg_str, 'outwordss.txt')


if __name__ == '__main__':
    main()

