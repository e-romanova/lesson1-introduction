"""Решение задания 1: создание, операции, индексация и формы тензоров."""
import torch


def main() -> None:
    torch.manual_seed(42)

    # 1.1 Создание тензоров
    random_3x4 = torch.rand(3, 4)
    zeros_2x3x4 = torch.zeros(2, 3, 4)
    ones_5x5 = torch.ones(5, 5)
    range_4x4 = torch.arange(16).reshape(4, 4)

    print("=== 1.1 Создание тензоров ===")
    print("Случайный 3x4:\n", random_3x4)
    print("Нули 2x3x4:\n", zeros_2x3x4)
    print("Единицы 5x5:\n", ones_5x5)
    print("Числа 0..15, форма 4x4:\n", range_4x4)

    # 1.2 Операции. A и B имеют заданные размеры.
    A = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    B = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3)
    print("\n=== 1.2 Операции ===")
    print("A:\n", A)
    print("B:\n", B)
    print("A.T (форма 4x3):\n", A.T)
    print("A @ B (матричное умножение, форма 3x3):\n", A @ B)
    print("A * B.T (поэлементно, форма 3x4):\n", A * B.T)
    print("Сумма элементов A:", A.sum().item())

    # 1.3 Индексация 3D тензора 5x5x5.
    cube = torch.arange(125).reshape(5, 5, 5)
    print("\n=== 1.3 Индексация (куб 5x5x5) ===")
    print("Первая строка первой матрицы cube[0, 0, :]:\n", cube[0, 0, :])
    print("Последний столбец каждой матрицы cube[:, :, -1]:\n", cube[:, :, -1])
    # Центральная 2x2 подматрица центрального слоя (слой с индексом 2).
    print("Центральная подматрица 2x2 центрального слоя:\n", cube[2, 1:3, 1:3])
    print("Элементы с чётными индексами по каждой оси cube[::2, ::2, ::2]:\n", cube[::2, ::2, ::2])

    # 1.4 Формы тензора из 24 элементов.
    values = torch.arange(24)
    print("\n=== 1.4 Преобразование форм ===")
    for shape in ((2, 12), (3, 8), (4, 6), (2, 3, 4), (2, 2, 2, 3)):
        print(f"Форма {shape}:\n{values.reshape(shape)}")


if __name__ == "__main__":
    main()
