import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 5, 3, 7, 9]

plt.plot(x, y)
plt.title("Exemplo de Gráfico de Linha")
plt.xlabel("Eixo X")
plt.ylabel("Eixo Y")
plt.plot(x, y, color="red", linestyle="--", marker="o")
plt.show()

for xi, yi in zip(x, y):
    plt.text(xi, yi, f"({xi}, {yi})")