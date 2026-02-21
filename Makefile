CC = cc
CFLAGS = -Wall -Wextra -O2
LDFLAGS = -lm

# ═══ BLAS Acceleration (optional) ═══
# Use: make BLAS=1
# macOS: Apple Accelerate (AMX/Neural Engine, zero deps)
# Linux: OpenBLAS (install: apt install libopenblas-dev)
#
# Without BLAS=1: pure scalar C loops (portable, correct)
# With BLAS=1: cblas_sgemv for delta voice, cblas_sger for notorch
UNAME := $(shell uname)

ifdef BLAS
  ifeq ($(UNAME), Darwin)
    CFLAGS += -DUSE_BLAS -DACCELERATE
    LDFLAGS += -framework Accelerate
  else
    CFLAGS += -DUSE_BLAS
    LDFLAGS += -lopenblas
  endif
endif

.PHONY: all test test-janus janus clean test-all test-blas

# ═══ Core AML ═══
all: libaml.a

libaml.a: core/ariannamethod.o
	ar rcs $@ $^

core/ariannamethod.o: core/ariannamethod.c core/ariannamethod.h
	$(CC) $(CFLAGS) -c $< -o $@

# ═══ AML Tests ═══
test: core/test_aml
	./core/test_aml

core/test_aml: core/test_aml.c core/ariannamethod.c core/ariannamethod.h
	$(CC) $(CFLAGS) core/test_aml.c core/ariannamethod.c -o $@ $(LDFLAGS)

# ═══ BLAS Tests — compile and run with acceleration ═══
test-blas:
ifeq ($(UNAME), Darwin)
	$(CC) $(CFLAGS) -DUSE_BLAS -DACCELERATE core/test_aml.c core/ariannamethod.c -o core/test_aml_blas -lm -framework Accelerate
else
	$(CC) $(CFLAGS) -DUSE_BLAS core/test_aml.c core/ariannamethod.c -o core/test_aml_blas -lm -lopenblas
endif
	./core/test_aml_blas

# ═══ Janus — first transformer in AML ═══
# "Janus will grow like mycelium, without roots, without a trunk, without a flag."

janus: janus/libjanus.dylib

janus/libjanus.dylib: janus/janus.go janus/lang.go janus/go.mod
	cd janus && go build -buildmode=c-shared -o libjanus.dylib .

test-janus: janus/libjanus.dylib janus/test_janus_c.c
	$(CC) $(CFLAGS) janus/test_janus_c.c -Ljanus -ljanus -o janus/test_janus_c
	cd janus && DYLD_LIBRARY_PATH=. ./test_janus_c

# Full test: AML + Janus basic
test-all: test test-janus

# ═══ Clean ═══
clean:
	rm -f core/*.o core/test_aml core/test_aml_blas libaml.a
	rm -f janus/libjanus.dylib janus/libjanus.h janus/test_janus_c
