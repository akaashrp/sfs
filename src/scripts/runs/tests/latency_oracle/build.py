"""Build the unmodified pinned upstream selector/cache with minimal type stubs.

No upstream algorithm is copied into test assertions. Offline build, no deps.
"""
import hashlib,json,os,shutil,subprocess
from pathlib import Path

def build(root, output, go):
    source=root/'src/assets/vllm_sr_latency'
    manifest=json.loads((source/'upstream.json').read_text())
    for name,digest in manifest['file_sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    output.mkdir(parents=True,exist_ok=True)
    def put(name,text):
        p=output/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
    put('go.mod','module github.com/vllm-project/semantic-router/src/semantic-router\n\ngo 1.20\n')
    put('pkg/config/types.go','package config\ntype ModelRef struct { Model string; LoRAName string }\n')
    put('pkg/observability/logging/stub.go','package logging\nfunc Warnf(s string,a ...interface{}) {}\nfunc Infof(s string,a ...interface{}) {}\nfunc Debugf(s string,a ...interface{}) {}\n')
    put('pkg/selection/stub.go','''package selection
import("context";"errors";"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config")
type SelectionMethod string
const MethodLatencyAware SelectionMethod = "latency_aware"
type SelectionContext struct { CandidateModels []config.ModelRef; LatencyAwareTPOTPercentile int; LatencyAwareTTFTPercentile int }
type SelectionResult struct { SelectedModel string; LoRAName string; Score float64; Confidence float64; Method SelectionMethod; Reasoning string; AllScores map[string]float64 }
type Feedback struct {}
func ValidateSelectionContext(s *SelectionContext) error { if s==nil || len(s.CandidateModels)==0 { return errors.New("empty") }; return nil }
func getModelNames(s []config.ModelRef) []string { r:=[]string{};for _,v:=range s {r=append(r,v.Model)};return r }
var _ = context.Background
''')
    for dest,name in [('pkg/selection/latency_aware.go','latency_aware.go'),('pkg/latency/cache.go','cache.go')]:
        put(dest,(source/name).read_text())
    put('main.go','''package main
import("context";"encoding/json";"os";"fmt";"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection";"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config";"github.com/vllm-project/semantic-router/src/semantic-router/pkg/latency")
type Event struct { Kind string; Model string; Metric string; Value float64; Models []string; TTFT int; TPOT int }
func main(){ var events []Event; if err:=json.NewDecoder(os.Stdin).Decode(&events);err!=nil {panic(err)}
 out:=[]*selection.SelectionResult{};for _,e:=range events { switch e.Kind {
 case "reset": latency.ResetTTFT();latency.ResetTPOT()
 case "update": if e.Metric=="ttft" {latency.UpdateTTFT(e.Model,e.Value)} else {latency.UpdateTPOT(e.Model,e.Value)}
 case "select": refs:=[]config.ModelRef{};for _,m:=range e.Models {refs=append(refs,config.ModelRef{Model:m})};r,err:=selection.NewLatencyAwareSelector(nil).Select(context.Background(),&selection.SelectionContext{CandidateModels:refs,LatencyAwareTPOTPercentile:e.TPOT,LatencyAwareTTFTPercentile:e.TTFT});if err!=nil {panic(err)};out=append(out,r)
 default:panic(fmt.Sprint(e)) }};json.NewEncoder(os.Stdout).Encode(out) }
''')
    env=dict(os.environ,GOCACHE=str(output/'cache'),GOTMPDIR=str(output/'scratch'),TMPDIR=str(output/'scratch'),GOTOOLCHAIN='local',GOPROXY='off',GOMAXPROCS='2')
    (output/'scratch').mkdir(exist_ok=True)
    subprocess.run([str(go),'build','-o','oracle','.'],cwd=output,env=env,check=True)
    return output/'oracle'
