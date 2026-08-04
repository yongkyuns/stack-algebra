MEMORY
{
  FLASH : ORIGIN = 0x00000000, LENGTH = 4M
  RAM : ORIGIN = 0x20000000, LENGTH = 4M
}

_stack_start = ORIGIN(RAM) + LENGTH(RAM);
_stack_end = _stack_start - 64K;
