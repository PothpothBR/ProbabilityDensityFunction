from scipy.integrate import quad
from numpy import dot, divide, power, sqrt, e, inf, subtract

# Função de densidade de probabilidade normal Z~N(0, 1)
def PDF_norm(x):
    return dot(
        0.39894228040143267793994605993438,
        power( e, dot( -0.5, power( x, 2 ) ) )
    )

# Normaliza V.A. de X~N(u, o2) para Z~N(0, 1)
def Z_score(u, o2): return divide( subtract( 1, u ), sqrt(o2) )

# Função normal padrão acumulada à esquerda P(Z <= z) ou A(z)
def Z_left_norm(z): return -round(quad(PDF_norm, inf, -z)[0], 4)

# Função normal padrão acumulada à direita P(Z >= z)
def Z_right_norm(z): return round(quad(PDF_norm, z, inf)[0], 4)

# Forma alternativa das funções normal padrão acumulada invertendo a aréa
def Z_left_norm2(z): return round(quad(PDF_norm, -z, inf)[0], 4)
def Z_right_norm2(z): return -round(quad(PDF_norm, inf, z)[0], 4)

# Forma alternativa das funções normal padrão acumulada usando a formula inversa
def Z_left_norm3(z): return 1-Z_right_norm(z)
def Z_right_norm3(z): return 1-Z_left_norm(z)

# atalho para as funções normal padrão acumulada
def Zl(z): return Z_left_norm(z)
def Zr(z): return Z_right_norm(z)

# atalho para a normalização da V.A.
def Z(u, o2): return Z_score(u, o2)

# atalhos para a função normal padrão acumulada com V.A. não normalizada
def Al(u, o2): return Zl(Z(u, o2))
def Ar(u, o2): return Zr(Z(u, o2))

# desenha a tabela para a função normal padrão acumulada
def print_Z_table(Z_norm):
    print("   ", end="")
    for Sz in range(0, 10):
        print("    {: .1f}".format(Sz), end="")
    print()

    print("      ", end="")
    for Sz in range(0, 10):
        print("________", end="")
    print()

    for Pz in range(-40, 41):
        Pz = Pz/10
        print("{: .1f}  |".format(Pz), end="")
        for Sz in range(0, 10):
            Sz = Sz/100
            z = Pz+(-Sz if Pz<0 else Sz)
            print("{: .4f}".format(Z_norm(z)), end=" ")
        print()

# Testando precisão usando dados conferidos como referência L_ e R_
print("L_: 0,6179", "R_: 0,3821")
print("L_: 0,3821", "R_: 0,6179")
print()
print("Lz:", Z_left_norm(0.3), "Rz:", Z_right_norm(0.3))
print("Lz:", Z_left_norm(-0.3), "Rz:", Z_right_norm(-0.3))
print()
print("L2:", Z_left_norm2(0.3), "R2:", Z_right_norm2(0.3))
print("L2:", Z_left_norm2(-0.3), "R2:", Z_right_norm2(-0.3))
print()
print("L3:", Z_left_norm2(0.3), "R3:", Z_right_norm2(0.3))
print("L3:", Z_left_norm2(-0.3), "R3:", Z_right_norm2(-0.3))
print()

# Exibindo as tabelas normal padrão acumulada
print("Left Table:")
print_Z_table(Zl)
print()
print("Right Table:")
print_Z_table(Zr)