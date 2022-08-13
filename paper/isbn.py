def is_isbn(isbn: str) -> bool:
    if not type(isbn) is str:
        return False
    if len(isbn) != 10:
        return False
    if not isbn[:9].isdigit():
        return False
    if not isbn[9] in "0123456789X":
        return False
    som = 0
    for i in range(9):
        cijfer = int(isbn[i])
        som += (i + 1) * cijfer
    if isbn[9] == "X":
        laatste_cijfer = 10
    else:
        laatste_cijfer = int(isbn[9])
    return laatste_cijfer == som % 11
