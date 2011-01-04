#!/usr/bin/env ruby

MAUDE = "/usr/local/maude/maude"
TMPDIR = "/tmp"

if not File.executable? MAUDE then
  puts "Error: maude exec not found at #{MAUDE}"
  exit
end

if not File.writable? TMPDIR then
  puts "Error: temp dir #{TMPDIR} is not writable"
  exit
end


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

tmp_file = File.join(TMPDIR, "guard_temp.maude")
tmph = File.new(tmp_file,"w")
tmph.puts "rew " + input_term + " .\nquit .\n"
tmph.close

maude_output = `#{MAUDE} -no-banner -interactive guard.maude #{tmp_file}`
md = /result Term:(.*)\nBye./m.match(maude_output)
result_term = md[1].split("\n").join(" ")

puts "Output term"
puts "-" * 78
puts result_term
puts

output = `echo '#{result_term}' | runghc TermToC.hs`
puts "Output file"
puts "-" * 78
puts output
puts
