require "option_parser"

struct Result
  getter :seed, :score

  def initialize(@seed : Int32, @score : Int64)
  end
end

NULL_RESULT = Result.new(0, -1i64)
SEED_BASE   = 1000
num_test = 1000
num_worker = 8
input_dir = "in"
ch = Channel(Result).new

OptionParser.parse do |parser|
  parser.on("-t NUM_TEST", "--tests=NUM_TEST", "Number of test cases(default:#{num_test})") do |v|
    num_test = v.to_i
  end
  parser.on("-w NUM_WORKER", "--workers=NUM_WORKER", "Number of workers(default:#{num_worker})") do |v|
    num_worker = v.to_i
  end
  parser.on("-i INPUT_DIR", "input directory(default:tools/in)") do |v|
    input_dir = v
  end
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
end

num_worker.times do |i|
  spawn do
    i.step(to: num_test - 1, by: num_worker) do |j|
      seed = sprintf("%04d", j + SEED_BASE)
      output = IO::Memory.new(1000)
      error = IO::Memory.new(1000)
      File.open("#{input_dir}/#{seed}.txt") do |f|
        Process.run("./main", input: f, output: output, error: error)
      end
      error.rewind
      m = error.gets_to_end.scan(/score:(\d+)/).last
      ch.send(Result.new(j + SEED_BASE, m[1].to_i64))
      # output.rewind
      # File.open("output/#{seed}.txt", "w") do |f|
      #   f << output.gets_to_end
      # end
    end
  end
end

results = Array.new(num_test, NULL_RESULT)
ri = 0
num_test.times do |i|
  res = ch.receive
  results[res.seed - SEED_BASE] = res
  while ri < results.size && results[ri].score != -1
    printf "seed:%04d score:%d\n", ri + SEED_BASE, results[ri].score
    ri += 1
  end
end
printf "ave:%.3f\n", results.sum { |r| r.score } / num_test
