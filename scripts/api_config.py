# API 配置文件
# Mimo API (token-plan)

API_KEY       = 'tp-shx3nrgd1una7e5mr0on0em8cz8m8j17skkwyohpnadbd3id'
API_URL       = 'https://token-plan-sgp.xiaomimimo.com/v1'
MODEL         = 'mimo-v2.5-pro'
TEMPERATURE   = 0.8  # 0.8最优 (0.3太确定易错, 0.8有足够随机性)
MAX_TOKENS    = 10000
SLEEP_SEC     = 0.3
TIMEOUT       = 120.0

# 关闭思考过程，减少 token 开销
DISABLE_THINKING = {"thinking": {"type": "disabled"}}
