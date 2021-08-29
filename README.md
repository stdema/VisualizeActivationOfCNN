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
