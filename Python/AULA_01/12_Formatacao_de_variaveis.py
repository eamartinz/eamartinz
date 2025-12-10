a = 'A'
b = 'B'
c = 1.1

formato = 'a={parametro1} b={parametro2} c={parametro3:.2f}'.format(
    parametro1=a, parametro2=b, parametro3=c)

print(formato)

nome = "Eduardo"
idade = 48
formato = '{n} tem {i} anos'
print(formato.format(n=nome, i=idade))

