from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dropout
from tensorflow.random import set_seed
from dga_classifier import data
import numpy as np

# 加载模型
loaded_model = load_model("lstm_model1.h5")

# 假设你有多个域名
# malicious_domains, malicious_labels = data.gen_malicious(10)
malicious_domains = [ 'kacftment', 'xjesop','oaiycar',  'xjesopcomp', 'bnezareisbad', 'uomlytrain', 'bbxzolice', 'cepjmonkeys', 'kyeyey', 'rfxhball', 'ykiiball', 'eqoaetball', 'oigckandfield', 'jlfndhockey', 'esbyball', 'pjkkerrari', 'weogkcheverolet', 'cwfyowelcamino', 'ygkgporsche', 'pjkkordf150', 'xxgelebmw330i', 'llebrulegacy', 'drioacivic', 'aoqetaprius', 'prkrwalk', 'myznment', 'aygfsign', 'rljlficlight', 'zziglane', 'amkainglane', 'rljlficjam', 'ikrcort', 'gglzay', 'amycageclaim', 'amkaengerjet', 'pjsga1008', 'kugdican765', 'gsgded8765', 'esqnhwest3456', 'nvghquerque', 'qcvorancisco', 'oatiiego', 'esosngeles', 'eysaork', 'trwznta', 'bpkeland', 'eywztle', 'wighingtondc', 'mjgd7f56qhyxibgpe4q', 'e0mhwx34i8kvif1', 'q4mfk67hati6epwn1pcvupk', 'wfqvq0utatk2ybmdyrk', 'qd1jmtibyt3p5ve', '3fmlql7n3hy87n3fmpo', '5betot72s2ubehg0ylqneta', 'shk2q6cvid70obmx3b3', 'yvi2ab1tin5x565', 'uxy2kde4ets0gb5', 'silowbkb', 'wpwrcfmdb', 'wupsymgoam', 'hskfjdmipvs', 'mehqyfjiarui', 'dqpmpdxlsaaai', 'uciwmhovghaeyu', 'flmkndpvgkohvuo', 'qexlnxdaclrftgds', 'fgxupetknbdinnesi', 'kgvcijtrbkabpawbol', 'gjhdtetccxypwimevpi', 'hcwteundawbuaunvsknu', 'hwbvylmhvcwrfohnibiyv', 'ayctfojxrduldwmnexcfno', 'embhsnwcqfhmboidqosueyp', 'pqcmlyxluclkuhpyuojpmwso', 'jxphuxiqgwfrkcwdskfadbods', 'wsikbubgdxlgropscsyenfviic', 'ayqrhwrinqavtwooynvhqtjadte', 'jidysxxueejwfsovjottegqofxmg', 'iglxbvkwxnnsxnsyexlnyhenqpnrl', 'paxmddgxutkpudacqutpkuifipaxmd', 'ybmpqjhdobvecafscgdovdldgajjgig', 'qigorcyjsjvkypz', 'wxkydlhaxigmnntatbm', 'haswcrllguostbodq', 'qzapzggsbefaseyb', 'hormxniez', 'rwmbayqxlbzuwhgythud', 'wdxumshzcig', 'mcrbhvveq', 'aabscjwqorhqewhkczrr', 'hzblxcjysxm', 'vswgwffevhu', 'rawdqjhduhho', 'zoipmnwr', 'nswzpxezznwv', 'uqzsaqjmy', 'ufpnvzlxhjky', 'iuhqhbmq', 'liblkcpub', 'dhycazbjhewd', 'iuhqhbmq', 'xdphxefo', 'mxlxxqfl', 'lslxvnmypitxftvl', 'uorlrrbtcwqk', 'nwcfedlvcsqhjech', 'wshilpvuaxl', 'annukbvkqtnn', 'iyqewbx', 'syabcvrtjcdx', 'fefigvolhs', 'krjhqithfvusvn', 'hakyjyz', 'folmecyca', 'igcowc', 'polscpta', 'gggoksmscm', 'tkhwvztyj', 'itiyphugt', 'jztuhpkhibll', 'vboavpvsquyl', 'ezflbdfuxtjj', 'eqgwufisehznfpttrjjgsbd', 'pejzsruurx', 'uydtzytzayfharzvg', 'igmlzgctzcdkqa', 'mjdrzedylsn', 'lpsxzqwgcoteaocklellorkh', 'zpqoikhwnihb', 'zzyckjqcazrxeydtf', 'rftiatgdtqfxsdp', 'kawkukoqcvmtpifvogwhx', 'myokiskk', 'skmgcwiqg', 'uoocswmyac', 'uosckyegkkg', 'iqiiuaweceim', 'eiscqsuciewgs', 'eiisggmukwwwwg', 'aagumkqauwymaaw', 'ywuqqmoeeysgcqiw', 'ceoagcscmysckewye', 'skegcmiiugwygmaeee', 'ocagquueywmqyyekwyy', 'kuokemymkayukksiiiiw', 'skkqwwqysaomceaigesko', 'ywumqoyqimyaouwiwkuyga', 'skuugcksmaoqwqesosyiikk', 'ywyyswomokayaosmamocsmsw', 'qgguiwkcgammyeiocioaueewg', 'yweiakooqugockiycgwscewkac', 'gmesswsuaickuouywiumoegyweg', 'kuqisacmcicmumayuqgucaewkuui', 'kusogkooaqesoygmkesiggiscggae', 'eicimukiksaecksekkqmweecmacemk', 'ceomiwmekocciiecmeycmskgqiyowem', 'nvghhstioohnsxalp', 'sxwfbuetnksf', 'onqvichwygsckxob', 'bwsiifavuoaluefol', 'apoeajknmpvrgu', 'ofwmshbouokqtnf', 'riicrcsrhomwf', 'pmpdagpioyx', 'aysburakvtwkb', 'ypebuhgggn', '', 't', 'wy', 'tul', 'volu', 'pujyc', 'lymeli', 'bonukim', 'lysoteri', 'foxanosum', 'viderucoqo', 'vicecycyrah', 'tuzizeluwaly', 'tucibupiraxuh', 'kymoxyryryfytu', 'kemidunekakowag', 'ciqirekymabecufo', 'hawybuwecipyxycil', 'zusityjacademicizu', 'nozacoconiwimyxadyw', 'xuqukonusizalakogomi', 'cicuvocylyqilyvucamov', 'xuqapyrogyfytoqapyjohu', 'ciqiqutuvanycerepugoriv'] 
malicious_labels = [ 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'banjori', 'corebot', 'corebot', 'corebot', 'corebot', 'corebot', 'corebot', 'corebot', 'corebot', 'corebot', 'corebot', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'cryptolocker', 'dircrypt', 'dircrypt', 'dircrypt', 'dircrypt', 'dircrypt', 'dircrypt', 'dircrypt', 'dircrypt', 'dircrypt', 'dircrypt', 'kraken', 'kraken', 'kraken', 'kraken', 'kraken', 'kraken', 'kraken', 'kraken', 'kraken', 'kraken', 'locky', 'locky', 'locky', 'locky', 'locky', 'locky', 'locky', 'locky', 'locky', 'locky', 'locky', 'pykspa', 'pykspa', 'pykspa', 'pykspa', 'pykspa', 'pykspa', 'pykspa', 'pykspa', 'pykspa', 'pykspa', 'qakbot', 'qakbot', 'qakbot', 'qakbot', 'qakbot', 'qakbot', 'qakbot', 'qakbot', 'qakbot', 'qakbot', 'ramdo', 'ramdo', 'ramdo', 'ramdo', 'ramdo', 'ramdo', 'ramdo', 'ramdo', 'ramdo', 'ramdo', 'ramdo', 'ramdo', 'ramdo', 'ramdo', 'ramdo', 'ramdo', 'ramdo', 'ramdo', 'ramdo', 'ramdo', 'ramdo', 'ramdo', 'ramdo', 'ramdo', 'ramnit', 'ramnit', 'ramnit', 'ramnit', 'ramnit', 'ramnit', 'ramnit', 'ramnit', 'ramnit', 'ramnit', 'simda', 'simda', 'simda', 'simda', 'simda', 'simda', 'simda', 'simda', 'simda', 'simda', 'simda', 'simda', 'simda', 'simda', 'simda', 'simda', 'simda', 'simda', 'simda', 'simda', 'simda', 'simda', 'simda', 'simda'] 


# benign_domains = data.get_alexa(len(malicious_domains), filename='dga_detection_dataset/benign_domain/top-1m.csv')
benign_domains = [ 'qq.com', 'yahoo.com', 'instagram.com','bilibili.com',  'zhihu.com', 'twitter.com', 'amazon.com', 'wikipedia.org', 'bing.com', 'csdn.net', 'linkedin.com', 'whatsapp.com', 'reddit.com', 'taobao.com', '163.com', 'sina.com.cn', 'live.com', 'microsoft.com', 'google.com.hk', 'github.com', 'zoom.us', 'office.com', 'yandex.ru', 'weibo.com', 'vk.com', 'jd.com', 'xvideos.com', 'tiktok.com', 'pornhub.com', 'canva.com', 'dzen.ru', 't.co', 'amazon.in', 'yahoo.co.jp', 'alipay.com', 'microsoftonline.com', 'aliexpress.com', 'naver.com', 'cnblogs.com', 'mail.ru', 'sohu.com', 'netflix.com', 'paypal.com', 'fandom.com', 'apple.com', 'stackoverflow.com', 'hao123.com', 'tmall.com', 'douban.com', 'xhamster.com', 'pinterest.com', '1688.com', 'flipkart.com', 'ebay.com', 'quora.com', 'spankbang.com', 'so.com', 'myshopify.com', 'spotify.com', 'imdb.com', 'adobe.com', 'msn.com', 'indeed.com', '360.cn', 'chaturbate.com', 'xnxx.com', 'telegram.org', 'aliyun.com', 'twimg.com', 'freepik.com', 'duckduckgo.com', 'youdao.com', 'google.co.in', 'deepl.com', 'etsy.com', 'twitch.tv', 'google.cn', 'amazon.co.uk', 'sogou.com', 'amazon.co.jp', 'feishu.cn', 'tencent.com', 'ilovepdf.com', 'mega.nz', 'douyu.com', 'imgur.com', 'wordpress.com', 'booking.com', 'instructure.com', 't.me', 'nih.gov', 'iqiyi.com', 'dropbox.com', 'medium.com', 'xiaohongshu.com', 'force.com', 'pixiv.net', 'smzdm.com', 'alibaba.com', 'soso.com', 'amazonaws.com', 'discord.com', 'avito.ru', 'tradingview.com', 'realsrv.com', 'youku.com', '3dmgame.com', 'slack.com', 'sciencedirect.com', 'bbc.com', 'grammarly.com', 'atlassian.net', 'w3schools.com', 'cnn.com', 'researchgate.net', 'ok.ru', 'redd.it', 'doubleclick.net', 'mediafire.com', 'wetransfer.com', 'nytimes.com', 'fiverr.com', 'gosuslugi.ru', 'eastmoney.com', 'runoob.com', 'onlyfans.com', 'digikala.com', 'tumblr.com', 'jianshu.com', 'trello.com', 'stripchat.com', 'behance.net', 'udemy.com', 'fc2.com', 'hupu.com', 'quizlet.com', 'espn.com', 'office365.com', 'toutiao.com', 'google.co.jp', 'amazon.it', 'ozon.ru', 'tistory.com', 'walmart.com', 'okta.com', 'vimeo.com', 'rakuten.co.jp', 'mozilla.org', 'huawei.com', 'speedtest.net', 'shopify.com', 'savefrom.net', 'shutterstock.com', 'douyin.com', 'docin.com', 'zoukankan.com', 'notion.so', 'chase.com', 'linktr.ee', 'ifeng.com', 'daum.net', 'google.ru', 'pconline.com.cn', 'figma.com', 'y2mate.com', 'investing.com', 'chinaz.com', 'hotstar.com', 'dcinside.com', 'chsi.com.cn', 'stackexchange.com', 'ali213.net', 'archive.org', 'godaddy.com', 'jb51.net', 'zol.com.cn', 'apple.com.cn', 'gamersky.com', 'pexels.com', 'amazon.ca', 'cloudfront.net', 'autohome.com.cn', 'zillow.com', 'upwork.com', 'binance.com', 'samsung.com', 'book118.com', 'theguardian.com', 'amazon.de', 'livedoor.com', 'unsplash.com'] 
# print(benign_domains, "\n\n")
benign_labels = ['benign']*len(benign_domains)

domains = benign_domains + malicious_domains 
labels =   benign_labels + malicious_labels


# 创建字符到整数的映射字典

char_to_int = {'1': 1, 'd': 2, '0': 3, 'x': 4, '4': 5, 'e': 6, 'n': 7, 'k': 8, 'y': 9, 'r': 10, 'a': 11, '9': 12, 's': 13, 't': 14, 'p': 15, 'w': 16, '.': 17, 'g': 18, '6': 19, 'q': 20, 'u': 21, 'z': 22, '8': 23, '7': 24, 'm': 25, '3': 26, 'f': 27, 'h': 28, 'j': 29, 'i': 30, 'v': 31, 'b': 32, 'o': 33, 'l': 34, '-': 35, '2': 36, '5': 37, 'c': 38}
#映射要与训练模型时一致
print(char_to_int)
max_feature = len(char_to_int) + 1

X = [[char_to_int[x] for x in domain] for domain in domains] #转换为整数序列
# print(X)
maxlen = np.max([len(x) for x in domains])  # 
X = sequence.pad_sequences(X, maxlen=maxlen)

y = [0 if x == 'benign' else 1 for x in labels]
print(y)

        
prediction = loaded_model.predict(X, batch_size=128)
res = [1 if x > .5 else 0 for x in prediction]
print(res)





