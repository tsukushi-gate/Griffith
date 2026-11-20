create_clock -name clk50 -period 20.000 [get_ports CLOCK_50]
derive_clock_uncertainty
set_false_path -from [get_ports {SW[*]}]
set_false_path -to   [get_ports {LEDR[*]}]
set_false_path -from [get_ports UART_RXD]
set_false_path -to   [get_ports UART_TXD]
