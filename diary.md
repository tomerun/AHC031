# AHC031

## 2024-03-23

これって空きが多めのケースだとコスト0が達成できてしまう？
そういう誰かがコスト0にしてるケースは自分がそうするのに失敗するとスコアほぼ0になってしまうから外せない。ここはしっかりやらないといけない

初日の壁構築コストが0なのをうまく使うべきのような気がする。

最初の解として、壁の移動は行わずに幅Wの部屋だけを作る方法をやってみた。
横に等間隔の線を引いて、その位置の移動で山登る。

サンプルに比べて4倍程度にしかなっていない。意外と悪いな。コスト0のケースは無し。そんな簡単じゃないか。

いや、初期解を等間隔にするのが全然だめで、i番目の部屋の平均広さに比例する広さにしたらだいぶ良くなった。
サンプルに比べてスコア20倍くらい。コスト0が1000ケース中5個ある。

逆に幅はW固定で壁を動かしまくる解法を考えてみると、面積コストは0にできるとして、壁コストの上界が `D*(N-1)*W*2` くらいになるから上の部屋完全固定解よりはいいのか。

さらに使う部屋の順番を小さいものから固定とすると、前の日の情報が与えられればDPで1日分だけの最小コストは出せそう。 => ビームサーチへの発展もあり

あと幅をWじゃなくて縦にいくつかに割って複数カラムにするという発展もある。

使う部屋の順番は、Nが小さかったら固定にしなくてもビットDPで最適解を出せる。

i番目の部屋のD日にわたるmax のN個の和をとってみると、1000 * 1000以下なのは1000ケース中14個しかなかった。コスト0達成可能なのは少なかった。なるほど


## 2024-03-24

DP/ビームサーチ的に決めていく場合、0日目をどうやって決めるのかという問題があって、  
D日進めたあと逆向きにまたDP/ビームサーチするのを反復するといいかも。


> 逆に幅はW固定で壁を動かしまくる解法を考えてみると、面積コストは0にできるとして、

実際どれくらい0にできるのか？　確かめてみる

部屋サイズをWの倍数に切り上げた和がD日全部 W * W 以下なのは、1000ケース中757個。なるほどそこまで多くはない。

250の倍数に切り上げると 936/1000, 100の倍数に切り上げると 993/1000

まずは一方向でDPするのを書いた。かなり望みがない部分の無駄な計算してそうだけどとりあえず雑に。  
わりかし良くなって見れる程度のスコアになった。ただ空きが小さいケースでは、昨日書いた幅W固定で壁を動かしまくる解法のコスト上界 `D*(N-1)*W*2` より10倍くらい悪くて要調査。  
面積の制約がだいぶきついんかなあ？

あと簡単ケースでコスト1が達成できるケースで取れてない。別途それ用の解法を作るべきか？

現状
```
seed:0000 score:4988 pena_area:0 pena_wall:4987
seed:0001 score:3058701 pena_area:1624700 pena_wall:1434000
seed:0002 score:456701 pena_area:50700 pena_wall:406000
seed:0003 score:87224 pena_area:0 pena_wall:87223
seed:0004 score:437187 pena_area:248900 pena_wall:188286
seed:0005 score:93785 pena_area:20400 pena_wall:73384
seed:0006 score:26729 pena_area:300 pena_wall:26428
seed:0007 score:31059 pena_area:0 pena_wall:31058
seed:0008 score:176785 pena_area:0 pena_wall:176784
seed:0009 score:6928 pena_area:0 pena_wall:6927
```


> D日進めたあと逆向きにまたDP/ビームサーチするのを反復するといいかも。

逆向きというより、中間の日は前後の状態両方を見て設定し直すのを繰り返すのがよさそう。実装は大変そうだが…


## 2024-03-25

とりあえず最後に各部屋をサイズ順に割り当て直すのをやっておく。  
1%くらいは意味あったけどこれ本体のアルゴリズムが甘いからだよねえ


```
seed:0000 score:    4287 pena_area:       0 pena_wall:    4286
seed:0001 score: 3058701 pena_area: 1624700 pena_wall: 1434000
seed:0002 score:  456701 pena_area:   50700 pena_wall:  406000
seed:0003 score:   87224 pena_area:       0 pena_wall:   87223
seed:0004 score:  421469 pena_area:  244500 pena_wall:  176968
seed:0005 score:   78385 pena_area:    5000 pena_wall:   73384
seed:0006 score:   26729 pena_area:     300 pena_wall:   26428
seed:0007 score:   28937 pena_area:       0 pena_wall:   28936
seed:0008 score:  176785 pena_area:       0 pena_wall:  176784
seed:0009 score:    6928 pena_area:       0 pena_wall:    6927
```

後ろ向きに反復するのすぐ書けそうだったので書いた。
```
seed:0000 score:    2975 pena_area:       0 pena_wall:    2974
seed:0001 score: 2492901 pena_area: 1070900 pena_wall: 1422000
seed:0002 score:  404001 pena_area:       0 pena_wall:  404000
seed:0003 score:   85167 pena_area:       0 pena_wall:   85166
seed:0004 score:  264917 pena_area:   86700 pena_wall:  178216
seed:0005 score:   88315 pena_area:       0 pena_wall:   88314
seed:0006 score:   16685 pena_area:       0 pena_wall:   16684
seed:0007 score:   30137 pena_area:       0 pena_wall:   30136
seed:0008 score:  179697 pena_area:       0 pena_wall:  179696
seed:0009 score:    4371 pena_area:       0 pena_wall:    4370
```

まあこんなもんかというところ。悪くなってるのもあるのは遅くなった分だろう。


## 2024-03-26

完全解狙うのをとりあえず作っておく。  
完全解じゃなく若干面積がはみ出る場合でも、壁動かさない解を初期解として作っておくのはありだろう。

＝> ありだろうと書いたが全然ありじゃなかった。壁固定は完全解用にしかならんわ  
まあでも1000ケース中完全解が可能な14ケースで全部スコア1を達成できてるのでこれはこれでよかろう。


逆に壁の移動コストを無視して面積だけできるだけ合わせるようにした解を作ってみたら、空きが少ないケースでこれまでの解答よりもだいぶ良くなった。  
壁を動かしまくる解法のコスト上界 `D*(N-1)*W*2` の3~4割のスコアになっていそう。  
偶然壁が重なって移動コストにならない分を考慮した厳密なスコア計算はしてないが…

こんな雑な解答でこれだけ出るのならまだまだ伸ばさないといけないということだよねえ

現状（スコアは正確でなく真のスコアよりpena_wallが若干悪い）
```
seed:0000 score:    2210 pena_area:       0 pena_wall:    2209
seed:0001 score:  472074 pena_area:       0 pena_wall:  472073
seed:0002 score:  263698 pena_area:       0 pena_wall:  263697
seed:0003 score:   92632 pena_area:    6400 pena_wall:   86231
seed:0004 score:  160450 pena_area:       0 pena_wall:  160449
seed:0005 score:   56586 pena_area:       0 pena_wall:   56585
seed:0006 score:   14452 pena_area:       0 pena_wall:   14451
seed:0007 score:   22592 pena_area:       0 pena_wall:   22591
seed:0008 score:  175285 pena_area:     300 pena_wall:  174984
seed:0009 score:    4364 pena_area:       0 pena_wall:    4363
```


## 2024-03-27

完全にD日を独立にやるのではなく、縦の壁の位置は固定にするのも試してみた。
これだけでも、空きが多いところ以外で全体的に2割くらい良くなった。これだけで良くなるということはまだまだ伸ばす余地がたくさんあるということだ

```
seed:0000 score:    2211 pena_area:       0 pena_wall:    2210
seed:0001 score:  242974 pena_area:       0 pena_wall:  242973
seed:0002 score:  133339 pena_area:       0 pena_wall:  133338
seed:0003 score:   97026 pena_area:       0 pena_wall:   97025
seed:0004 score:   67431 pena_area:       0 pena_wall:   67430
seed:0005 score:   62871 pena_area:       0 pena_wall:   62870
seed:0006 score:   14452 pena_area:       0 pena_wall:   14451
seed:0007 score:   23009 pena_area:       0 pena_wall:   23008
seed:0008 score:   94079 pena_area:       0 pena_wall:   94078
seed:0009 score:    4371 pena_area:       0 pena_wall:    4370
```
