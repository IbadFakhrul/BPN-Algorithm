# **PERHITUNGAN MANUAL BACKPROPAGATION NEURAL NETWORK**

Jaringan terdiri dari 4 data, 2 fitur (input layer), 2 hidden layer, dan 1 output layer, dan nilai laju pembelajaran (α) = 0.01

Stopping condition = iterasi yang dilakukan = 1

Training data:

|  x1  |  x2  |  y   |
| :--: | :--: | :--: |
|  1   |  1   |  0   |
|  1   |  0   |  1   |
|  0   |  1   |  1   |
|  0   |  0   |  0   |

Bobot Input - Hidden

|      | z1         | z2         |
| ---- | ---------- | ---------- |
| x0   | v01 = 0.44 | v02 = 0.25 |
| x1   | v11 = 0.24 | v12 = 0.43 |
| x2   | v21 = 0.34 | v22 = 0.1  |

Bobot Hidden - Output

|      | y1         |
| ---- | ---------- |
| z0   | w01 = 0.56 |
| z1   | w11 = 0.34 |
| z2   | w21 = 0.95 |



**Iterasi Pertama Data Training ke-1**

**A.** Hitung keluaran hidden layer dengan rumus:
$$
z\_in_j = v_{0j} + \sum_{i=1}^{n}x_iv_{ij}
$$
$$
\\
z\_net_1=0.44+1*0.24+1*0.34=1.02 \\
z\_net_2=0.25+1*0.43+1*0.10=0.78
$$

**B.** tentukan sinyal output dari hidden unit di atas, dengan rumus:
$$
z_j=f(z\_net_j)=\dfrac{1}{1+e^{-z\_net_j}}
$$
$$
\\
z_1=\dfrac{1}{1+e^{-1.02}}=0.734972599 \\
z_2=\dfrac{1}{1+e^{-0.78}}=0.685680114
$$
**C.** Lakukan hal yang sama untuk variabel output (y) menggunakan sinyal output dari hidden unit dengan rumus:
$$
y\_net_k = w_{0k} + \sum_{j=1}^{p}z_jw_{jk}
$$
​	Dan:
$$
y_k=f(y\_net_k)=\dfrac{1}{1+e^{-y\_net_k}}
$$
$$
\\
y\_net_1=0.56+0.734972599*0.34+0.685680114*0.95=1.6478
\\
y_1=\dfrac{1}{1+e^{-1.6478}}=0.838593492
$$

**D.** Hitung delta bobot antara hidden dengan output, karena ini training data pertama maka target (t) = 0, dan memperbarui bobot Hidden dengan Output (bobot w)
$$
\delta_k=(t_k-y_k)f'(y\_net_k)=(tk-y_k)y_k(1-y_k) \\
\\
\delta_1=(0-0.838593492)*0.838593492*(1-0.838593492)=-0.113507358
$$
​	Pembaruan bobot, seperti disebutkan di awal bahwa (α) = 0.01:
$$
\Delta{w_{jk}}=\alpha\delta_{k}z_j\\
\\
\Delta{w01}=0.01*-0.113507358=-0.001135074 \\
\Delta{w11}=0.01*-0.113507358*0.734972599=-0.000834248 \\
\Delta{w21}=0.01*-0.113507358*0.685680114=-0.000778297
$$
**E.** Menjumlahkan input delta yang dikirim dari dari layer di langkah sebelumnya yang sudah berbobot
$$
\delta\_net_j=\sum_{k=1}^{m}\delta_kw_{jk}\\
\\
\delta\_net_1=-0.113507358*0.34=-0.038592502 \\
\delta\_net_2=-0.113507358*0.95=-0.10783199
$$
**F.** Hitung delta bobot antara input dengan hidden, dan memperbarui bobot Input dengan Hidden (bobot v)
$$
\delta_j=\delta\_net_jf'(z\_net_j)=\delta\_net_jz_j(1-z_j) \\
\\
\delta_1=-0.038592502*0.734972599*(1-0.734972599)=-0.007517352 \\
\delta_2=-0.10783199*0.685680114*(1-0.685680114)=-0.023240263
$$
​	Pembaruan bobot:
$$
\Delta{v_{ij}}=\alpha\delta_{j}z_j\\
\\
\Delta{v_{01}}=0.01*-0.007517352=-0.00007517 \\
\Delta{v_{11}}=0.01*-0.007517352*0.734972599=-0.00005525 \\
\Delta{v_{21}}=0.01*-0.007517352*0.685680114=-0.00005154 \\
\Delta{v_{02}}=0.01*-0.023240263=-0.00023240 \\
\Delta{v_{12}}=0.01*-0.023240263*0.734972599=-0.00017081 \\
\Delta{v_{22}}=0.01*-0.023240263*0.685680114=-0.000159354
$$
**G.** Update nilai bobot

​	Hidden - Output:
$$
w_{jk}(2)=w_{jk}+\Delta{w_{jk}}\\
\\
w_{01}(2)=0.56+(-0.001135074)=0.558864926 \\
w_{11}(2)=0.34+(-0.000834248)=0.339165752 \\
w_{21}(2)=0.95+(-0.000778297)=0.949221703
$$
​	Input - Hidden:
$$
v_{ij}(2)=v_{ij}+\Delta{v_{ij}}\\
\\
v_{01}(2)=0.44+(-0.00007517)=0.43992483 \\
v_{11}(2)=0.24+(-0.00005525)=0.23994475 \\
v_{21}(2)=0.34+(-0.00005154)=0.33994846 \\
v_{02}(2)=0.25+(-0.00023240)=0.24976760 \\
v_{12}(2)=0.43+(-0.00017081)=0.42982919 \\
v_{22}(2)=0.1+(-0.000159354)=0.09984065 \\
$$
**H.** Lakukan semua langkah di atas untuk setiap data training yang tersedia, dengan bobotnya adalah bobot yang baru saja di update di training data sebelumnya, sehingga ditemukan hasilnya:

​	Data training ke-2:
$$
w_{01}=0.55916386\\
w_{11}=0.339364157\\
w_{21}=0.94942009\\\\
	
v_{01}=0.43994746\\
v_{11}=0.23995977\\
v_{21}=0.33996347\\
v_{02}=0.24983094\\
v_{12}=0.42987123\\
v_{22}=0.09988268
$$
​	Data training ke-3:
$$
w_{01}=0.559599839\\
w_{11}=0.339663091\\
w_{21}=0.949675812\\\\
	
v_{01}=0.43997935\\
v_{11}=0.23998163\\
v_{21}=0.33998218\\
v_{02}=0.24993132\\
v_{12}=0.42994006\\
v_{22}=0.09994156
$$
​	Data training ke-4:
$$
w_{01}=0.55814877\\
w_{11}=0.338780472\\
w_{21}=0.94886008\\\\
	
v_{01}=0.43986190\\
v_{11}=0.23991020\\
v_{21}=0.33991616\\
v_{02}=0.24959213\\
v_{12}=0.42973375\\
v_{22}=0.09975088
$$


Karena iterasi pertama sudah selesai maka stopping condition sudah terpenuhi
