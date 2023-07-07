import torch

full_size = 4

A = torch.ones((full_size, full_size))
B = torch.ones((full_size, full_size))

C = A @ B

A00 = A[:2, :2]
A01 = A[:2, 2:]
A10 = A[2:, :2]
A11 = A[2:, 2:]

B00 = B[:2, :2]
B01 = B[:2, 2:]
B10 = B[2:, :2]
B11 = B[2:, 2:]

A00 = A00.to("cuda:0")
B00 = B00.to("cuda:0")
C00 = A00 @ B00
A00 = A00.to("cuda:1")
B01 = B01.to("cuda:1")
C01 = A00 @ B01
A10 = A10.to("cuda:2")
B00 = B00.to("cuda:2")
C10 = A10 @ B00
A10 = A10.to("cuda:3")
B01 = B01.to("cuda:3")
C11 = A10 @ B01

A01 = A01.to("cuda:0")
B10 = B10.to("cuda:0")
C00 += A01 @ B10
A01 = A01.to("cuda:1")
B11 = B11.to("cuda:1")
C01 += A01 @ B11
A11 = A11.to("cuda:2")
B10 = B10.to("cuda:2")
C10 += A11 @ B10
A11 = A11.to("cuda:3")
B11 = B11.to("cuda:3")
C11 += A11 @ B11

C = C.to("cuda:0")
print(torch.allclose(C[:2, :2], C00))
C = C.to("cuda:1")
print(torch.allclose(C[:2, 2:], C01))
C = C.to("cuda:2")
print(torch.allclose(C[2:, :2], C10))
C = C.to("cuda:3")
print(torch.allclose(C[2:, 2:], C11))

print("2d tensor parallel forward example finish")
