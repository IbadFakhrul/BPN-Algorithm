# **PERHITUNGAN MANUAL BACKPROPAGATION NEURAL NETWORK**

Jaringan terdiri dari 4 data, 2 fitur (input layer), 2 hidden layer, dan 1 output layer, iterasi dihentikan jika eror mencapai 0.8

Training data:

|  x1  |  x2  |  y   |
| :--: | :--: | :--: |
|  1   |  1   |  0   |
|  1   |  0   |  1   |
|  0   |  1   |  1   |
|  0   |  0   |  0   |

Bobot

|      | Nilai |
| ---- | ----- |
| v01  | 0.3   |
| v11  | 0.1   |
| v21  | 0.3   |
| v02  | 0.9   |
| v12  | -0.1  |
| v22  | 0.2   |
| w01  | 0.4   |
| w11  | -0.7  |



Dengan bobot di atas, tentukan error untuk training data secara keseluruhan dengan Mean Square Error, dengan rumus:
$$
z\_in_j = v_{0j} + \sum_{i=1}^{n}x_iv_{ij}
$$


Maka didapat:


$$
z\_in_{11}=0.3+(1*0.1)+(1*0.3)=0.7\\
z\_in_{12}=0.3+(1*0.1)+(0*0.3)=0.4\\
z\_in_{13}=0.3+(0*0.1)+(1*0.3)=0.6\\
z\_in_{14}=0.3+(0*0.1)+(0*0.3)=0.3\\
z\_in_{12}=0.9+(1*-0.1)+(1*0.2)=1\\
z\_in_{22}=0.9+(1*-0.1)+(0*0.2)=0.8\\
z\_in_{32}=0.9+(0*-0.1)+(1*0.2)=1.1\\
z\_in_{42}=0.9+(0*-0.1)+(0*0.2)=0.9
$$


Lalu tentukan sinyal output dari hidden unit di atas, dengan rumus:
$$
Z_j=f(z\_in_j)=\dfrac{1}{1+e^{-z\_in_j}}
$$
Maka didapat:
$$
z_{11}=0.331812228\\
z_{21}=0.40131234\\
z_{31}=0.354343694\\
z_{41}=0.425557483\\
z_{12}=0.268941421\\
z_{22}=0.310025519\\
z_{32}=0.249739894\\
z_{42}=0.289050497
$$
Lakukan hal yang sama untuk variabel output (y) menggunakan sinyal output dari hidden unit dengan rumus:
$$
z\_in_k = v_{0k} + \sum_{j=1}^{p}z_jw_{jk}
$$
Didapat:
$$
y\_in_{11}=0.4+(0.331812228*-0.7)=0.16773144
\\
y\_in_{21}=0.4+(0.40131234*-0.7)=0.119081362
\\
y\_in_{31}=0.4+(0.249739894*-0.7)=0.225182074
\\
y\_in_{41}=0.4+(0.289050497*-0.7)=0.197664652
$$
Maka error:
$$
E=0.5*((0-0.16773144)^2+(1-0.119081362)^2+(1-0.225182074)^2+(0-0.197664652)^2)=0.721782808
$$
Karena eror kurang dari 0.8 maka stepping condition sudah terpenuhi, dan kasus sudah selesai.
