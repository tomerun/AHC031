OBJ = main.o
TARGET = main
# COMPILEOPT = -Wall -Wextra -Wshadow -Wno-sign-compare -std=gnu++17 -O2 -DLOCAL -D_GLIBCXX_DEBUG -g
COMPILEOPT = -Wall -Wextra -Wshadow -Wno-sign-compare -std=gnu++20 -mtune=native -march=native -fconstexpr-depth=2147483647 -fconstexpr-loop-limit=2147483647 -fconstexpr-ops-limit=2147483647 -O2 -DLOCAL  
vpath %.cpp ..
vpath %.h ..

.PHONY: all clean

all: $(TARGET)

# $(TARGET): main.cpp
# 	clang++ -std=c++17 -Wall -Wextra -Wno-sign-compare -O2 -DONLINE_JUDGE -DATCODER -mtune=native -march=native -fconstexpr-depth=2147483647 -fconstexpr-steps=2147483647 -DLOCAL -o $@ $<

$(TARGET): $(OBJ)
	g++-13 -o $@ $(OBJ)

%.o: %.cpp main.cpp
	g++-13 $(COMPILEOPT) -c $<

clean:
	rm -f $(TARGET)
	rm -f $(OBJ)
