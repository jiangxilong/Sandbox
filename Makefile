CC = g++-4.4
CFLAGS = -g -Wall `pkg-config opencv --cflags `
LIBS = `pkg-config opencv --libs` -lfreenect -lpthread
OBJS = kineccv.o
PROG = kinectopencv

all: $(PROG)

$(PROG):  $(OBJS)
	$(CC) $(LIBS) $(OBJS) -o $(PROG)

kineccv.o: kineccv.cpp
	$(CC) $(CFLAGS)  $(LIBS) -c $<

clean:
	rm -f $(OBJS) $(PROG)
