# from googletrans import Translator

# translator = Translator(service_urls=['translate.google.cn'])
# text = '用支付宝付款时失败了'
# res_1 = translator.translate(text, dest='english') # 可选其他english\French
# res_2 = translator.translate(res_1.text, dest='zh-cn')
# print(res_2)

import jionlp as jio
xunfei_api = jio.XunfeiApi(
        [{"appid": "5f5846b1",
          "api_key": "52465bb3de9a258379e6909c4b1f2b4b",
          "secret": "b21fdc62a7ed0e287f31cdc4bf4ab9a3"}])
google_api = jio.GoogleApi()
baidu_api = jio.BaiduApi(
        [{'appid': '20200618000498778',
          'secretKey': 'raHalLakgYitNuzGOoB2'},  # 错误的密钥
         {'appid': '20200618000498778',
          'secretKey': 'raHalLakgYitNuzGOoBZ'}], gap_time=0.5)

apis = [baidu_api, google_api, xunfei_api]
back_trans = jio.BackTranslation(mt_apis=apis)
text = '用支付宝付款时失败了'
result = back_trans(text)
print(result)
