# Discord_Bot(.심리 상담사)
**해당 봇은 친구들의 음성 데이터셋을 확보하기 위해 만들졌습니다.**

사용하기 쉽도록 하나의 파이썬 파일로 뭉쳐놨습니다.

---
## 사용을 위해 .env파일에 Client Secret key, OpenAI API key, Gemini API key를 작성해야합니다.

또한 스트리밍 녹음을 위해서는 Google OAuth 2.0 클라이언트정보가 담긴 json이 필요하며

[Google OAuth 2.0를 위함 링크](https://console.cloud.google.com/apis/credentials?inv=1&invt=AblxRA&project=velvety-glyph-297514)

구글 드라이브 파일 ID를 바꿔줘야 합니다.

![image](https://github.com/user-attachments/assets/d7e8b739-0dfd-41de-bfe6-cc8c05318523)

이런 형식으로 구글 드라이브에 업로드가 됩니다.

스트리밍 녹을을 원치 않으시면 Not Stream_record를 쓰시면 됩니다.

---
## 부수적으로 TTS기능과 LLM 기능을 추가 하였습니다.
사용하기 앞서 /basics_channel을 사용하여 tts 채널과 LLM 채널을 지정해줘야 합니다

#### - TTS기능
- 토대 채널에서 작성하는 글을 봇이 읽어줍니다.
  
- !를 사용하여 말을 중단 시키거나, TTS기능을 수행 안 시킬수 있습니다.
> !앞의 내용을 중단 시키고 송신자의 이름을 발설 안합니다.
> 해당 말은 TTS를 사용 안 합니다!

- <>를 사용하여 TTS에 색다르게 꾸밀수 있습니다.
> 1. <2.0>이렇게 숫자를 넣으면 <0.5>속도가 조절이 됩니다.
> 2. <+2.0>이렇게 +숫자를 넣으면 <+0.5>피치가 조절이 됩니다.
> 3. <-1000>이렇게 -숫자를 넣으면 페이드 인 시킬수 있습니다.
> 4. <!2.0>이렇게 !숫자를 넣으면 <!0.5>데시벨이 조절이 됩니다.
> 5. re는 역재생을 rm는 글자 애너그램을
> 6. ja와 같이 gTTS에서 제공하는 언어를 넣으면 언어의 종류를 변경시킬수 있습니다.

- 사운드보드 기능을 사용할 수 있습니다.
> ![image](https://github.com/user-attachments/assets/062465be-59a3-44b6-9695-799b4be23f0c)
> ![image](https://github.com/user-attachments/assets/3157748a-f195-45f7-ad9e-5bcb8c5e3803)
> ![image](https://github.com/user-attachments/assets/ca3396bd-730c-4ec5-b9de-b2dcd8f8c095)
> 이런 형식으로 저장 및 사용 할 수 있습니다.

- 사진을 올리면 설명을 해줍니다.
> ![image](https://github.com/user-attachments/assets/0e9c0e9c-52ad-4ecf-9a65-c66ca8c12a0a)
> 이런 형식으로 Gemini가 짧은 분석을 하고 TTS로 답변을 해줍니다.

#### - LLM기능
- 토대 채널에서 작성하는 글을 선택한 봇이 api를 거져 답변을 해줍니다.
