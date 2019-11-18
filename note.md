### Attack Method

ok 目前的攻击方法是L∞攻击，可以尝试改成L2攻击, ∞更好

ok transformer 后面尝试改成没有esim，效果可以即可

ok tranformer 没有第一个lstm encoder效果是变好还是变差,应该是变差

ok 准则函数变成soft label

攻击方式改成Jacabian 删掉最重要的一个词， 两个句子都删掉最重要的一个词

使用attention进行攻击

一种基于attention，bert的在文本匹配领域的黑盒攻击方式，
如果可以的话可以在论文里写提升了GAAA的效果

cifar10,fine_tuning ,no finetuning, all finetuning尝试都可以写在论文中

ok cifar10十个类别的脆弱性图

vulnerability 加在哪儿的比较

ok 不同epsilon的效果比较

ok 不同criterion loss的效果比较

两阶段loss曲线下降图、vulnerability加在哪里是怎么判断的，为了最大化利用预训练的效果

1.FGSM不能完全捕获脆弱信息，只是一种度量方式
2.对于不同数据集，有不同的攻击方式能够获取更为合适的脆弱性度量
3.cifar10效果不好原因有三，类多，干扰信息，已经训练的过拟合
4.分析别的数据集是否biased

Q3和Q4需要改变

新的融入方式的架构图和loss图

模仿figure 1新截取部分图，加上是否可以被反向传播

解释一下为啥用F1，F2

需要强调model A and model B share the same model framework, 一开始B的weights也是共享的A的

GAAA 把脆弱性信息encoding进句子