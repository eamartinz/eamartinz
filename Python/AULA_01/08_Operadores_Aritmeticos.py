# Operadores aritméticos em Python
# Python suporta os seguintes operadores aritméticos básicos:
# +  : Adição
# -  : Subtração
# *  : Multiplicação
# /  : Divisão
# // : Divisão inteira (floor division)
# %  : Módulo (resto da divisão)
# ** : Exponenciação (potência)

# Exemplo de uso dos operadores aritméticos
soma = 10 + 10
print('Soma =', soma)

subtracao = 10 - 5
print('Subtração =', subtracao)

multiplicacao = 10 * 10
print('Multiplicação =', multiplicacao)

divisao = 10 / 2.2
print('Divisão =', divisao)

divisao_inteira = 10 // 2.2
print('Divisao inteira =', divisao_inteira)

resto_da_divisao = 25 % 3
print('Resto da divisão =', resto_da_divisao)

exponenciacao = 2 ** 10
print('Exponenciação =', exponenciacao)

# Exemplo: vamos checar se doi números são múltiplos::
numero = 10
divisor = 8
print('O número', numero, ' é divisível por', divisor, '?', numero % divisor == 0)

numero = 16
print('O número', numero, ' é divisível por', divisor, '?', numero % divisor == 0)

# Exemplo: vamos testar se um número é par ou ímpar
numero = 42
print('O número', numero, ' é par?', numero % 2 == 0)
numero = 33
print('O número', numero, ' é par?', numero % 2 == 0)