#!/usr/bin/env python3
"""API测试脚本"""
import requests, time

API_KEY = 'sk-cp-JstUWpAJpyIJBq9PRbmeaby_BUpj-Gqj6zXiXyCWevAU4coQCHp6WLvmWrEBHcwW1njIBhGAJH96A06_6asltqnw1pdqLkOZSn78Ym5xBQ8cFAD8om5csOc'
URL = 'https://api.minimaxi.com/v1/text/chatcompletion_v2'
MODEL = 'MiniMax-M2.7-highspeed'

def test_api(n=5, prompt="Hello, how are you?"):
    success = 0
    times = []
    errors = []
    
    for i in range(n):
        headers = {'Authorization': 'Bearer ' + API_KEY}
        data = {
            'model': MODEL,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 50000,
            'temperature': 0.7
        }
        
        start = time.time()
        try:
            r = requests.post(URL, json=data, headers=headers, timeout=60)
            elapsed = time.time() - start
            times.append(elapsed)
            
            if r.status_code == 200:
                resp = r.json()
                content = resp.get('choices', [{}])[0].get('message', {}).get('content', '')
                if content and len(content) > 5:
                    success += 1
                    print('  [' + str(i+1) + '/' + str(n) + '] ✅ ' + str(round(elapsed,1)) + 's | content: ' + content[:50])
                else:
                    print('  [' + str(i+1) + '/' + str(n) + '] ⚠️  ' + str(round(elapsed,1)) + 's | empty')
            else:
                print('  [' + str(i+1) + '/' + str(n) + '] ❌ HTTP ' + str(r.status_code))
                errors.append('HTTP ' + str(r.status_code))
        except Exception as e:
            elapsed = time.time() - start
            print('  [' + str(i+1) + '/' + str(n) + '] ❌ ' + str(round(elapsed,1)) + 's | ' + str(e))
            errors.append(str(e))
        
        time.sleep(2)
    
    print('\n📊 测试结果: ' + str(success) + '/' + str(n) + ' 成功')
    if times:
        avg = sum(times) / len(times)
        print('   平均响应时间: ' + str(round(avg,1)) + 's')
        print('   最快: ' + str(round(min(times),1)) + 's, 最慢: ' + str(round(max(times),1)) + 's')
    if errors:
        print('   错误: ' + str(errors[:3]))

if __name__ == '__main__':
    print('🧪 MiniMax API 测试开始...\n')
    test_api(n=5)
