import pyopencl as cl
import numpy
import numpy.linalg as la

a = (numpy.random.rand(100000, 110)*1).astype(numpy.float32)
c = numpy.zeros((1, 110)).astype(numpy.float32)
	
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

prg = cl.Program(ctx, """
    __kernel void sum(__global const float (*a)[110],
    __global float c[110])
    {
      int gid = get_global_id(0);
      c[gid] = 0;
      for (int i = 0; i < 100000; i++)
        c[gid] += a[i][gid] ;
    }
    """).build()

#kernel = prg.sum

prg.sum(queue, (c.shape[1], ), None, a_buf, dest_buf)

gid = (numpy.empty(c.shape)).astype(numpy.float32)
cl.enqueue_read_buffer(queue, dest_buf, gid).wait()

print numpy.allclose(a.sum(0), gid)
