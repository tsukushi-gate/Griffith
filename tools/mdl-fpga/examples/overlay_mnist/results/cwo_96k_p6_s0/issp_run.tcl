# mdl-fpga 自動生成: ISSP で overlay を走らせて probe を読む
set hw  [lindex [get_hardware_names] 0]
set dev [lindex [get_device_names -hardware_name $hw] 0]
puts "HW=$hw DEV=$dev"
start_insystem_source_probe -device_name $dev -hardware_name $hw
proc rd {} {
    set v [read_probe_data -instance_index 0 -value_in_hex]
    regsub {^0x} $v {} v
    return [expr wide(0x$v)]
}
proc wr {val} {
    write_source_data -instance_index 0 -value $val -value_in_hex
}
wr 2 ; after 50 ; wr 0 ; after 50 ; wr 1
set done 0 ; set t 0
while {$t < 60000} {
    set v [rd] ; set done [expr $v & 1]
    if {$done} break
    after 20 ; incr t 20
}
set v [rd]
puts "PROBE_RAW=$v"
puts "PROBE_WAIT_MS=$t"
wr 0
end_insystem_source_probe
