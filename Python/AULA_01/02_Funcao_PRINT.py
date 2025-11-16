# A função print é usada para imprimir mensagens na tela.
# Podemos usar a função print para exibir textos, números e outros tipos de dados.

print("Olá, Mundo!")  # Imprime a mensagem "Olá, Mundo!" na tela.
print(42)             # Imprime o número 42 na tela.
print(3.14)           # Imprime o número decimal 3.14 na tela.
print(True)          # Imprime o valor booleano True na tela.

print(12, 34, 56)  # Imprime múltiplos valores separados por vírgulas.

print(12, 34, 56, sep=' - ')  # Imprime múltiplos valores com um separador personalizado.

# Caracteres especiais na função print
# \r - Retorno de carro
# \n - Nova linha
# \t - Tabulação
print("Linha 1\nLinha 2\nLinha 3")  # Imprime em múltiplas linhas.
print("Coluna 1\tColuna 2\tColuna 3")  # Imprime com tabulações entre colunas.
print("Início\rFim")  # O texto "Fim" sobrescreve "Início" devido ao retorno de carro.

# Modificar o fim da impressão com o parâmetro 'end'
print("Primeira linha.", end=' ')
print("Continua na mesma linha.")
print("Segunda linha.", end=' *** ')
print("Termina aqui.")

# Python reconhece a diferença entre minúsculas e maiúsculas
Print("Isto causará um erro.")  # Isso causará um erro porque 'Print' não é a forma correta da função, que deve ser 'print'.
# Corrigindo o erro:
print("Isto funcionará corretamente.")  # Agora está correto.