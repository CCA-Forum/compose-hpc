#!/usr/bin/env ruby

tmp_dir = ENV['TMPDIR']
input_file = ARGV[0]

puts "Input file"
puts "-" * 78
puts `cat #{input_file}`
puts

input_term = `runghc CToTerm.hs #{input_file}`.strip
puts "Input term"
puts "-" * 78
puts input_term
puts

tmp_file = tmp_dir + "guard_temp.maude"
tmph = File.new(tmp_file,"w")
tmph.puts "rew " + input_term + " .\nquit .\n"
tmph.close

maude_output = `maude -no-banner -interactive guard.maude #{tmp_file}`
md = /result Term:(.*)\nBye./m.match(maude_output)
result_term = ""
md[1].each_line do |line|
  result_term += line.strip
end

puts "Output term"
puts "-" * 78
puts result_term
puts

output = `echo '#{result_term}' | runghc TermToC.hs`
puts "Output file"
puts "-" * 78
puts output
puts
