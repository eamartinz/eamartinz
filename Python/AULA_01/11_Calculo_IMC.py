# Programa para calcular o Índice de Massa Corporal (IMC)
# Fórmula do IMC: IMC = peso / (altura ** 2)

nome = 'Eduardo'
altura = 1.88
peso = 75

imc = peso / (altura * altura)

print(nome, "tem", altura, "de altura.")
print("Pesa", peso, "kg e seu IMC é de:")
print(imc)