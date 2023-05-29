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
y\_net_1=0.56+1*0.34+1*0.95=1.85
\\
y_1=\dfrac{1}{1+e^{-1.85}}=0.864127103
$$

**D.** Hitung delta bobot antara hidden dengan output, karena ini training data pertama maka target (t) = 0, dan memperbarui bobot Hidden dengan Output (bobot w)
$$
\delta_k=(t_k-y_k)f'(y\_net_k)=(tk-y_k)y_k(1-y_k) \\
\\
\delta_1=(0-0.864127103)*0.864127103*(1-0.864127103)=-0.101458419\\
$$
​	Pembaruan bobot, seperti disebutkan di awal bahwa (α) = 0.01:
$$
\Delta{w_{jk}}=\alpha\delta_{k}z_j\\
\\
\Delta{w01}=0.01*-0.101458419=-0.001014584 \\
\Delta{w11}=0.01*-0.101458419*0.734972599=-0.000745692 \\
\Delta{w21}=0.01*-0.101458419*0.685680114=-0.00069568
$$
**E.** Menjumlahkan input delta yang dikirim dari dari layer di langkah sebelumnya yang sudah berbobot
$$
\delta\_net_j=\sum_{k=1}^{m}\delta_kw_{jk}\\
\\
\delta\_net_1=-0.101458419*0.34=-0.0344959 \\
\delta\_net_2=-0.101458419*0.95=-0.0963855
$$
**F.** Hitung delta bobot antara input dengan hidden, dan memperbarui bobot Input dengan Hidden (bobot v)
$$
\delta_j=\delta\_net_jf'(z\_net_j)=\delta\_net_jz_j(1-z_j) \\
\\
\delta_1=-0.0344959*0.734972599*(1-0.734972599)=-0.006719383 \\
\delta_2=-0.0963855*0.685680114*(1-0.685680114)=-0.020773282
$$
​	Pembaruan bobot:
$$
\Delta{v_{ij}}=\alpha\delta_{j}x_i\\
\\
\Delta{v_{01}}=0.01*-0.006719383=-0.00006719 \\
\Delta{v_{11}}=0.01*-0.006719383*1=-0.00006719 \\
\Delta{v_{21}}=0.01*-0.006719383*1=-0.00006719 \\
\Delta{v_{02}}=0.01*-0.020773282=-0.000207733 \\
\Delta{v_{12}}=0.01*-0.020773282*1=-0.000207733 \\
\Delta{v_{22}}=0.01*-0.020773282*1=-0.000207733
$$
**G.** Update nilai bobot

​	Hidden - Output:
$$
w_{jk}(2)=w_{jk}+\Delta{w_{jk}}\\
\\
w_{01}(2)=0.56+-0.001014584=0.558985416 \\
w_{11}(2)=0.34+-0.000745692=0.339254308 \\
w_{21}(2)=0.95+-0.00069568=0.94930432
$$
​	Input - Hidden:
$$
v_{ij}(2)=v_{ij}+\Delta{v_{ij}}\\
\\
v_{01}(2)=0.44+(-0.00006719)=0.43993281 \\
v_{11}(2)=0.24+(-0.00006719)=0.23993281 \\
v_{21}(2)=0.34+(-0.00006719)=0.33993281 \\
v_{02}(2)=0.25+(-0.000207733)=0.249792267 \\
v_{12}(2)=0.43+(-0.000207733)=0.429792267 \\
v_{22}(2)=0.1+(-0.000207733)=0.099792267 \\
$$
**H.** Lakukan semua langkah di atas untuk setiap data training yang tersedia, dengan bobotnya adalah bobot yang baru saja di update di training data sebelumnya, sehingga ditemukan hasilnya:

​	Data training ke-2:
$$
w_{01}=0.559284254\\
w_{11}=0.33945265\\
w_{21}=0.949502643\\\\
	
v_{01}=0.43995543\\
v_{11}=0.23995543\\
v_{21}=0.33995543\\
v_{02}=0.24985559\\
v_{12}=0.42979227\\
v_{22}=0.09979227
$$
​	Data training ke-3:
$$
w_{01}=0.559720146\\
w_{11}=0.339751524\\
w_{21}=0.949758307\\\\
	
v_{01}=0.43998733\\
v_{11}=0.23995543\\
v_{21}=0.33995543\\
v_{02}=0.24995596\\
v_{12}=0.42989264\\
v_{22}=0.09989264
$$
​	Data training ke-4:
$$
w_{01}=0.558269126\\
w_{11}=0.338868932\\
w_{21}=0.948942593\\\\
	
v_{01}=0.43986986\\
v_{11}=0.23995543\\
v_{21}=0.33995543\\
v_{02}=0.24961676\\
v_{12}=0.42989264\\
v_{22}=0.09989264
$$


Karena iterasi pertama sudah selesai maka stopping condition sudah terpenuhi
