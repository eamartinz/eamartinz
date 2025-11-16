# Python classifica as variáveis em diferentes tipos de dados.
"""
Os tipos de dados mais comuns em Python são:
- int: Números inteiros (ex: 10, -5, 0)
- float: Números decimais (ex: 3.14, -0.001, 2.0)
- str: Cadeias de caracteres (ex: "Olá", 'Python', "123")
- bool: Valores booleanos (True ou False)
- list: Listas (ex: [1, 2, 3], ["a", "b", "c"])
- tuple: Tuplas (ex: (1, 2, 3), ("x", "y", "z"))
- dict: Dicionários (ex: {"chave": "valor", "idade": 25})
- set: Conjuntos (ex: {1, 2, 3}, {"a", "b", "c"})
"""

# Strings são textos delimitados por aspas simples ou duplas.
print("Eduardo")  # Tipo str
print('Python é divertido!')  # Tipo str

nome = "Eduardo"  # Variável do tipo str
print(nome)

# Para imprimir caracteres especiais em strings, usamos a barra invertida (\).
print("Linha 1\nLinha 2")  # Nova linha
print("Coluna 1\tColuna 2")  # Tabulação
print("C:\\Users\\Eduardo")  # Barra invertida
print("Ele disse: \"Olá!\"")  # Aspas duplas dentro de aspas duplas
print('Ela respondeu: \'Oi!\'')  # Aspas simples dentro de aspas simples

# Posso imprimir aspas dentro de strings usando aspas diferentes para delimitar a string.
print("Ele disse: 'Olá!'")  # Aspas simples dentro de aspas duplas
print('Ela respondeu: "Oi!"')  # Aspas duplas dentro de aspas simples

# Para usar expressões regulares em strings, podemos usar o prefixo 'r' antes das aspas.
print(r"C:\Users\Eduardo\Documents")  # String raw, ignora caracteres especiais

# Para imprimir múltiplas linhas, podemos usar aspas triplas.
print("""Esta é a linha 1.
Esta é a linha 2.
Esta é a linha 3.""")  # Múltiplas linhas com aspas triplas
print('''Outra forma de múltiplas linhas.
Usando aspas simples triplas.''')  # Múltiplas linhas com aspas simples triplas

