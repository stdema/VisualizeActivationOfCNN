# VisualizeActivationOfCNN
 - 이 예제에서는 컨벌루션 신경망에 영상을 입력하고 신경망의 여러 계층의 활성화 결과를 표시하는 방법을 보여준다.

원본 영상과 활성화 영역을 비교해보면 신경망이 어떤 특징을 학습했는지 알아낼 수 있다. 앞쪽 계층의 채널은 색이나 경계같은 단순한 특징을 학습하는 반면 더 깊은 계층의 채널은 눈과 같은 복잡한 특징을 학습한다는 것을 알 수 있다.

### 사전 훈련된 신경망과 데이터 불러오기
```c
net = squeezenet;

im = imread('face.jpg');
imshow(im)

imgSize = size(im);
imgSize = imgSize(1:2);
```
사전 훈련된 신경망 [squeezenet]

![image](https://user-images.githubusercontent.com/86040099/131237756-601209de-77c1-4f67-a8b9-44efcab8100b.png)

나중에 이미지 사이즈 사용을 위해 저장

### 신경망 구조 보기
```c
analyzeNetwork(net)
```

![image](https://user-images.githubusercontent.com/86040099/131237792-8deadb0e-be01-4371-b516-f4fc537b89ba.png)

### 첫 번째 컨벌루션 계층의 활성화 표시하기

이미지를 신경망에 통과시킨 후 conv1 계층의 출력 활성화 결과를 살펴본다.

```c
act1 = activations(net, im, 'conv1');

sz = size(act1);
act1 = reshape(act1, [sz(1) sz(2) 1 sz(3)]);

I = imtile(mat2gray(act1), 'GridSize', [8 8]);
imshow(I)
```

![image](https://user-images.githubusercontent.com/86040099/131237842-f7876baa-a96e-4c2d-8e3c-f9dcbaddf890.png)

### 특정 채널의 활성화 조사하기

```c
act1ch22 = act1(:,:,:,22);
act1ch22 = mat2gray(act1ch22);
act1ch22 = imresize(act1ch22, imgSize);

I = imtile({im, act1ch22});
imshow(I);
```

![image](https://user-images.githubusercontent.com/86040099/131237931-7952dafc-79ad-4b91-91cb-460a0aaf683a.png)

밝은 흰색이 빨간색 영역에 대응되므로 이 채널은 빨간색 픽셀에서 활성화됨.

### 활성화가 가장 강한 채널 찾기

```c
[maxValue, maxValueIndex] = max(max(max(act1)));
act1chMax = act1(:,:,:,maxValueIndex);
act1chMax = mat2gray(act1chMax);
act1chMax = imresize(act1chMax, imgSize);

I = imtile({im, act1chMax});
imshow(I)
```

![image](https://user-images.githubusercontent.com/86040099/131237978-2ef5a706-f915-4d73-834c-ce778aa702e3.png)

원본과 비교하여 이 채널이 경계에서 활성화되는 것을 알 수 있음. 왼쪽이 밝고 오른쪽이 어두운 경계를 양으로 활성화함.

### 심층 계층 조사하기
대부분의 컨벌루션 신경망은 심층 컨벌루션 계층에서 더 복잡한 특징을 검출하도록 학습한다. 뒤의 계층은 이전 계층의 특징들을 조합하여 특징을 구축한다. 'conv1'과 같은 방식으로 'fire6-squeeze1x1'계층을 조사한다.

```c
act6 = activations(net,im,'fire6-squeeze1x1');
sz = size(act6);
act6 = reshape(act6,[sz(1) sz(2) 1 sz(3)]);

I = imtile(imresize(mat2gray(act6),[64 64]),'GridSize',[6 8]);
imshow(I)
```

![image](https://user-images.githubusercontent.com/86040099/131238059-cf706c16-62e0-493a-b788-56e2fb7d8210.png)

이미지가 많으므로 fire6-squeeze1x1 계층에서 활성화가 가장 강한 채널 표시

```c
[maxValue6,maxValueIndex6] = max(max(max(act6)));
act6chMax = act6(:,:,:,maxValueIndex6);
imshow(imresize(mat2gray(act6chMax),imgSize))
```

![conv1](https://user-images.githubusercontent.com/86040099/131245348-54ebcd3f-cffc-4cd1-999b-f471da0fa443.jpg)

여기서는 가장 강한 활성화 채널이 다른 채널에 비해 흥미로운 세부 특징을 보이지 않으며, 양의 활성화(밝은 부분)뿐 아니라 음의 활성화(어두운 부분) 반응도 강하게 나타나고 있음.
-그리드에 표시된 모든 채널 중 눈에서 활성화하는 채널이 있을 수 있음. 확인을 위해 채널 14와 채널 47을 조사

```c
I = imtile(imresize(mat2gray(act6(:,:,:,[14 47])),imgSize));
imshow(I)
```

![conv1-1](https://user-images.githubusercontent.com/86040099/131245492-334a0046-0425-4425-a291-e25252a4f4f4.jpg)

-양의 활성화 값만 조사하려면 분석을 반복하여 fire6-relu_squeeze1x1 계층의 활성화 결과를 시각화

```c
act6relu = activations(net,im,'fire6-relu_squeeze1x1');
sz = size(act6relu);
act6relu = reshape(act6relu,[sz(1) sz(2) 1 sz(3)]);

I = imtile(imresize(mat2gray(act6relu(:,:,:,[14 47])),imgSize));
imshow(I)
```

![conv1-2](https://user-images.githubusercontent.com/86040099/131245571-02818359-0a41-44f5-917f-78d4dfff3eb1.jpg)

fire6-squeeze1x1 계층의 활성화 결과와 비교해 보면 fire6-relu_squeeze1x1 계층의 활성화 결과는 영상에서 얼굴 특징이 강한 영역을 명확하게 찾아내고 있음.

### 채널이 눈을 인식하는지 테스트하기
-fire6-relu_squeeze1x1 계층의 채널 14와 47이 눈에서 활성화되는지 확인

-신경망에 한쪽 눈이 감긴 새 영상을 입력하고 이 영상의 활성화 결과를 원본 영상의 활성화와 비교

```c
imClosed = imread('face-eye-closed.jpg');
imshow(imClosed)
```
![conv1-3](https://user-images.githubusercontent.com/86040099/131245627-7513f45b-19b5-42e1-b68f-bb4cfcfc8c64.jpg)

```c
act6Closed = activations(net,imClosed,'fire6-relu_squeeze1x1');
sz = size(act6Closed);
act6Closed = reshape(act6Closed,[sz(1),sz(2),1,sz(3)]);
```

원래 영상과 해당 활성화 결과를 하나의 Figure에 플로팅

```c
channelsClosed = repmat(imresize(mat2gray(act6Closed(:,:,:,[14 47])),imgSize),[1 1 3]);
channelsOpen = repmat(imresize(mat2gray(act6relu(:,:,:,[14 47])),imgSize),[1 1 3]);
I = imtile(cat(4,im,channelsOpen*255,imClosed,channelsClosed*255));
imshow(I)
title('Input Image, Channel 14, Channel 47');
```

![conv1-4](https://user-images.githubusercontent.com/86040099/131245666-16912069-d022-4fcd-8f04-da2589aecc4d.jpg)

-채널 14와 47 모두 양쪽 눈에서 활성화되었으며 입 주변도 어느 정도 활성화되었음을 확인할 수 있음.

-신경망은 눈을 학습하라는 지시를 받은 적이 없지만, 여러 영상 클래스를 구분하는 데 눈이 유용한 특징임을 학습했다.

-기존의 머신러닝 방식에서는 종종 문제에 맞게 특징을 수동으로 설계했지만 심층 컨벌루션 신경망은 스스로 유용한 특징을 학습할 수 있음을 알 수 있음.
