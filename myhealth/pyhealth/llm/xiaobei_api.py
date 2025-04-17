from openai import OpenAI


class XiaoBei:
    def __init__(self) -> None:
        openai_api_key = "EMPTY"
        openai_api_base = "http://123.57.228.132:8285/v1"

        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

    def request(self, input_text: str) -> str:
        query = input_text
        chat_response = self.client.chat.completions.create(
            model="qwen/Qwen1.5-32B-Chat",
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": query},
            ],
            temperature=0.95,
            max_tokens=4096
        )
        response = chat_response.choices[0].message.content.strip()
        return response


if __name__ == '__main__':
    text = '大夫，中成药可以吃壮腰健肾丸吗？\n1.经常嘴唇干，稍微干活就口中粘腻，脖子有些僵硬.\n2走路容易累和腿脚软，挺腰也感觉无力.\n3.去年夏天看过医生大夫说我阴液大伤肾阴不足，脾胃虚弱，吃梨爱拉肚子，从小就是吃了有半年中药有很好的转变眼睛睡醒干涩偶尔睡觉出汗一模没汗脖子就是很凉\n5我还需要继续吃柳氮磺吡啶等西药吗舌根部有少，就是问问能吃壮腰健肾丸和柴胡疏肝散吗'
    model = XiaoBei()
    answer = model.request(text)
    print(answer)