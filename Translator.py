import requests
import urllib

def translate_line(line, source_lang, dest_lang):
    parsed_text = urllib.parse.quote(line, safe='')
    URI = "https://translate.googleapis.com/translate_a/single?client=gtx&sl={}&tl={}&dt=t&q={}".format(source_lang,dest_lang,parsed_text)
    r = requests.get(URI)
    return eval(r.text.replace("null", "0"))[0][0][0]