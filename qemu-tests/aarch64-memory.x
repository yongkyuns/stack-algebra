ENTRY(_start)

MEMORY
{
  RAM (rwx) : ORIGIN = 0x40080000, LENGTH = 127M
}

SECTIONS
{
  .text :
  {
    . = ALIGN(16);
    KEEP(*(.text._start))
    *(.text .text.*)
  } > RAM

  .rodata : ALIGN(16)
  {
    *(.rodata .rodata.*)
  } > RAM

  .data : ALIGN(16)
  {
    *(.data .data.*)
  } > RAM

  .bss (NOLOAD) : ALIGN(16)
  {
    __bss_start = .;
    *(.bss .bss.*)
    *(COMMON)
    __bss_end = .;
  } > RAM

  .stack (NOLOAD) : ALIGN(16)
  {
    __stack_bottom = .;
    . += 64K;
    __stack_top = .;
  } > RAM
}
