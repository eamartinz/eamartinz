# -------------------------------------------------------------
# Teste rápido (executa quando usar: python dataset.py)
# -------------------------------------------------------------
if __name__ == "__main__":
    loader = criar_dataloaders()

    print("Total de imagens:", len(loader.dataset))

    # Pega um único batch
    for imgs, labels in loader:
        print("Batch de imagens:", imgs.shape)
        print("Batch de labels:", labels)
        break