import { type CSSProperties, type PointerEvent as ReactPointerEvent, useEffect, useRef, useState } from 'react';
import {
  type AgentEvent,
  type BrainType,
  type CaseTemplate,
  type LLMStatusResponse,
  type LocalExecutionDevice,
  type LocalModelProfile,
  type ModelToolMetricsResponse,
  type RunResponse,
  createAgent,
  createCaseTemplate,
  createCaseRun,
  createRun,
  createWorkflowRun,
  executeRun,
  getCases,
  getHealth,
  getLLMStatus,
  getLocalModels,
  getModelToolMetrics,
  activateLocalModel,
  clearModelToolMetrics,
  getReplay,
  openRunEventStream,
  pauseRun,
  resumeRun,
  stepRun,
} from './api';
import AgentEditor, { type WorkflowDocument } from './AgentEditor';

const panelStyle: CSSProperties = {
  border: '1px solid #d9dee8',
  borderRadius: 12,
  padding: 18,
  background: '#ffffff',
};

const terminalStatuses = new Set(['completed', 'failed', 'waiting_user']);

function formatEventData(event: AgentEvent): string {
  return JSON.stringify(event.data, null, 2);
}

type FlowNodeKind = 'start' | 'context' | 'brain' | 'route' | 'tool' | 'observation' | 'wait' | 'finish' | 'error';

interface FlowNode {
  id: string;
  kind: FlowNodeKind;
  title: string;
  subtitle: string;
  step: number;
  contextDepth?: number;
}

interface FlowEdge {
  from: string;
  to: string;
  label?: string;
}

interface CycleFrame {
  step: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

interface Point {
  x: number;
  y: number;
}

function stringValue(value: unknown, fallback = ''): string {
  if (typeof value === 'string') return value;
  if (value === undefined || value === null) return fallback;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function percentage(value: number | null): string {
  return value === null ? '—' : `${(value * 100).toFixed(1)}%`;
}

const decisionReasonLabels: Record<string, string> = {
  connection: '连接', protocol: '协议/JSON', action_validation: '动作校验',
  route: '路由', capability: '能力边界', workflow: '图结构', other: '其他',
};

function decisionBreakdownText(breakdown: Record<string, number>): string {
  const items = Object.entries(breakdown).filter(([, count]) => count > 0);
  return items.length ? items.map(([kind, count]) => `${decisionReasonLabels[kind] || kind} ${count}`).join(' · ') : '—';
}

function actionLabel(event: AgentEvent | undefined): string {
  const action = (event?.data.action ?? {}) as Record<string, unknown>;
  const actionType = stringValue(action.action, '模型处理中');
  if (actionType === 'call_tool') return 'call_tool · ' + stringValue(action.tool, 'tool');
  if (actionType === 'call_tools') return `call_tools · ${Array.isArray(action.tool_calls) ? action.tool_calls.length : 0} 个工具`;
  if (actionType === 'final_answer') return 'final_answer';
  if (actionType === 'ask_user') return 'ask_user';
  if (actionType === 'route') return 'route · ' + stringValue(action.target, 'next Agent');
  return actionType;
}

function flowNodeHeight(node: FlowNode): number {
  return node.kind === 'context' ? Math.min(166, 94 + Math.max(0, (node.contextDepth ?? 0) - 3) * 12) : 94;
}

function buildFlow(events: AgentEvent[], run: RunResponse | null): { nodes: FlowNode[]; edges: FlowEdge[] } {
  const nodes: FlowNode[] = [
    {
      id: 'run-start',
      kind: 'start',
      title: 'Run start',
      subtitle: run?.input || '等待运行',
      step: 0,
    },
  ];
  const edges: FlowEdge[] = [];
  const steps = [...new Set(events.filter((event) => event.step > 0).map((event) => event.step))].sort(
    (left, right) => left - right,
  );
  let previous = 'run-start';
  let previousContextDepth = 0;

  for (const step of steps) {
    const contextEvent = events.find((event) => event.step === step && event.type === 'context.build');
    const requestEvent = events.find((event) => event.step === step && event.type === 'context.request');
    const contextMessages = Array.isArray(requestEvent?.data.messages)
      ? requestEvent.data.messages
      : Array.isArray(contextEvent?.data.messages) ? contextEvent.data.messages : [];
    if (contextEvent) {
      const contextId = `context-${step}`;
      nodes.push({
        id: contextId,
        kind: 'context',
        title: `Context · step ${step}`,
        subtitle: `${contextMessages.length} 条消息${previousContextDepth ? ` · 本轮新增 ${Math.max(0, contextMessages.length - previousContextDepth)} 条` : ''}${Array.isArray(requestEvent?.data.tools) ? ` · ${requestEvent?.data.tools.length} 个工具定义` : ''}${step > 1 ? ' · 已注入历史轨迹' : ''}`,
        step,
        contextDepth: contextMessages.length,
      });
      edges.push({ from: previous, to: contextId, label: previous === 'run-start' ? '用户输入注入 Context' : undefined });
      previous = contextId;
      previousContextDepth = contextMessages.length;
    }

    const llmEnd = events.find((event) => event.step === step && event.type === 'llm.end');
    const llmStarted = events.some((event) => event.step === step && event.type === 'llm.start');
    const brainId = `brain-${step}`;
    nodes.push({
      id: brainId,
      kind: 'brain',
      title: `LLM Think · step ${step}`,
      subtitle: llmEnd ? actionLabel(llmEnd) : llmStarted ? '模型处理中' : '未开始',
      step,
    });
    edges.push({ from: previous, to: brainId, label: contextEvent ? '组装完成 → 进入 LLM 推理' : undefined });
    previous = brainId;

    const routed = events.find((event) => event.step === step && event.type === 'agent.route');
    if (routed) {
      const routeId = `route-${step}`;
      nodes.push({
        id: routeId,
        kind: 'route',
        title: 'Route to Agent',
        subtitle: stringValue(routed.data.target, '选择下一位 Agent'),
        step,
      });
      edges.push({ from: previous, to: routeId });
      previous = routeId;
    }

    const toolStarts = events.filter((event) => event.step === step && event.type === 'tool.start');
    if (toolStarts.length > 0) {
      const toolId = `tool-${step}`;
      const isParallel = toolStarts.length > 1 && toolStarts.some((event) => event.data.parallel === true);
      nodes.push({
        id: toolId,
        kind: 'tool',
        title: toolStarts.length === 1 ? `Tool · ${toolStarts[0].node}` : `Tool batch · ${toolStarts.length} calls`,
        subtitle: toolStarts.length === 1 ? stringValue(toolStarts[0].data.args, '执行工具') : `${isParallel ? '并行' : '串行'} · ${toolStarts.map((event) => event.node).join(' + ')}`,
        step,
      });
      edges.push({ from: previous, to: toolId });
      previous = toolId;

      const toolEnds = events.filter((event) => event.step === step && event.type === 'tool.end');
      if (toolEnds.length > 0) {
        const observationId = `observation-${step}`;
        nodes.push({
          id: observationId,
          kind: 'observation',
          title: 'Observation',
          subtitle: toolEnds.length === 1
            ? `${stringValue(toolEnds[0].data.result, '工具已返回')} · ${stringValue(toolEnds[0].data.call_id, 'call_id 未知')}`
            : `${toolEnds.length} 条 Observation 已写回 Context · ${toolEnds.map((event) => stringValue(event.data.call_id, event.node)).join('、')}`,
          step,
        });
        edges.push({ from: previous, to: observationId });
        previous = observationId;
      }
    }

    const confirmation = events.find((event) => event.step === step && event.type === 'agent.confirmation_required');
    const waiting = events.find((event) => event.step === step && event.type === 'agent.waiting_user');
    if (confirmation) {
      const waitId = `confirm-${step}`;
      nodes.push({ id: waitId, kind: 'wait', title: 'User confirmation', subtitle: stringValue(confirmation.data.calls, '等待高风险工具确认'), step });
      edges.push({ from: previous, to: waitId });
      previous = waitId;
    }
    if (waiting) {
      const waitId = `wait-${step}`;
      nodes.push({
        id: waitId,
        kind: 'wait',
        title: 'Ask user',
        subtitle: stringValue(waiting.data.question, '等待用户输入'),
        step,
      });
      edges.push({ from: previous, to: waitId });
      previous = waitId;
    }

    const finish = events.find((event) => event.step === step && event.type === 'agent.finish');
    if (finish) {
      const finishId = `finish-${step}`;
      nodes.push({
        id: finishId,
        kind: 'finish',
        title: 'Final answer',
        subtitle: stringValue(finish.data.output, run?.output || '完成'),
        step,
      });
      edges.push({ from: previous, to: finishId });
      previous = finishId;
    }

    const failure = events.find(
      (event) => event.step === step && (event.type === 'agent.error' || event.type === 'llm.error' || event.type === 'tool.error'),
    );
    if (failure) {
      const errorId = `error-${step}`;
      nodes.push({
        id: errorId,
        kind: 'error',
        title: 'Error',
        subtitle: stringValue(failure.data.error, '执行失败'),
        step,
      });
      edges.push({ from: previous, to: errorId });
      previous = errorId;
    }
  }

  return { nodes, edges };
}

const flowColors: Record<FlowNodeKind, { background: string; border: string; label: string }> = {
  start: { background: '#eaf4ff', border: '#5a9ee8', label: '#1e5c99' },
  context: { background: '#f2efff', border: '#9383df', label: '#5545a2' },
  brain: { background: '#eaf8f1', border: '#54ad7a', label: '#18794e' },
  route: { background: '#f5edff', border: '#9a6fd1', label: '#68429b' },
  tool: { background: '#fff5e5', border: '#e1a64e', label: '#8a5a00' },
  observation: { background: '#fff9df', border: '#d4b63f', label: '#776000' },
  wait: { background: '#fff8e6', border: '#e1a64e', label: '#8a5a00' },
  finish: { background: '#e9f8f0', border: '#38a169', label: '#176b43' },
  error: { background: '#fff0ef', border: '#d45d52', label: '#a33a2b' },
};

function FlowCanvas({
  events,
  run,
  selectedNodeId,
  onSelect,
}: {
  events: AgentEvent[];
  run: RunResponse | null;
  selectedNodeId: string;
  onSelect: (nodeId: string) => void;
}) {
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<{ id: string; offsetX: number; offsetY: number } | null>(null);
  const flow = buildFlow(events, run);
  const nodeKey = flow.nodes.map((node) => node.id).join('|');
  const [positions, setPositions] = useState<Record<string, Point>>({});
  const [viewportWidth, setViewportWidth] = useState(0);

  useEffect(() => {
    setPositions((current) => {
      const next = { ...current };
      const steps = [...new Set(flow.nodes.filter((node) => node.step > 0).map((node) => node.step))];
      if (!next['run-start']) next['run-start'] = { x: 28, y: 128 };
      let cursorX = 230;
      steps.forEach((step) => {
        const stepNodes = flow.nodes.filter((node) => node.step === step);
        stepNodes.forEach((node, index) => {
          if (!next[node.id]) next[node.id] = { x: cursorX + index * 192, y: 104 };
        });
        cursorX += Math.max(390, stepNodes.length * 192 + 42);
      });
      const known = new Set(flow.nodes.map((node) => node.id));
      Object.keys(next).forEach((id) => {
        if (!known.has(id)) delete next[id];
      });
      return next;
    });
  }, [nodeKey]);

  useEffect(() => {
    const viewport = canvasRef.current?.parentElement;
    if (!viewport) return;
    const update = () => setViewportWidth(viewport.clientWidth);
    update();
    const observer = new ResizeObserver(update);
    observer.observe(viewport);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    function move(event: PointerEvent) {
      const drag = dragRef.current;
      const canvas = canvasRef.current;
      if (!drag || !canvas) return;
      const rect = canvas.getBoundingClientRect();
      setPositions((current) => ({
        ...current,
        [drag.id]: {
          x: Math.max(8, event.clientX - rect.left - drag.offsetX),
          y: Math.max(8, event.clientY - rect.top - drag.offsetY),
        },
      }));
    }
    function stop() {
      dragRef.current = null;
    }
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', stop);
    return () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', stop);
    };
  }, []);

  const canvasWidth = Math.max(
    1120,
    viewportWidth,
    ...flow.nodes.map((node) => (positions[node.id]?.x || 0) + 216),
  );
  const canvasHeight = Math.max(330, ...flow.nodes.map((node) => (positions[node.id]?.y || 0) + flowNodeHeight(node) + 48));
  const cycleFrames: CycleFrame[] = [...new Set(flow.nodes.filter((node) => node.step > 0).map((node) => node.step))]
    .map((step) => {
      const stepNodes = flow.nodes.filter((node) => node.step === step);
      const points = stepNodes.map((node) => positions[node.id]).filter((point): point is Point => Boolean(point));
      if (!points.length) return null;
      const left = Math.min(...points.map((point) => point.x));
      const right = Math.max(...points.map((point) => point.x + 168));
      const top = Math.min(...points.map((point) => point.y));
      const bottom = Math.max(...stepNodes.map((node) => (positions[node.id]?.y || 0) + flowNodeHeight(node)));
      return { step, x: left - 18, y: top - 56, width: right - left + 36, height: bottom - top + 76 };
    })
    .filter((frame): frame is CycleFrame => Boolean(frame));
  const flowNodeMap = new Map(flow.nodes.map((node) => [node.id, node]));

  function startDrag(event: ReactPointerEvent, node: FlowNode) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const position = positions[node.id] || { x: 8, y: 8 };
    dragRef.current = {
      id: node.id,
      offsetX: event.clientX - rect.left - position.x,
      offsetY: event.clientY - rect.top - position.y,
    };
  }

  return (
    <div style={{ overflowX: 'auto', border: '1px solid #e4e8f0', borderRadius: 10, background: '#fbfcff' }}>
      <div ref={canvasRef} style={{ position: 'relative', width: canvasWidth, height: canvasHeight }}>
        <svg width={canvasWidth} height={canvasHeight} style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
          <defs>
            <marker id="flow-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 z" fill="#9aa5b5" />
            </marker>
          </defs>
          {cycleFrames.map((frame) => (
            <g key={`cycle-${frame.step}`}>
              <rect x={frame.x} y={frame.y} width={frame.width} height={frame.height} rx="18" fill="#f6f2ff" />
              <path
                d={`M ${frame.x + frame.width} ${frame.y} H ${frame.x + 18} Q ${frame.x} ${frame.y} ${frame.x} ${frame.y + 18} V ${frame.y + frame.height - 18} Q ${frame.x} ${frame.y + frame.height} ${frame.x + 18} ${frame.y + frame.height} H ${frame.x + frame.width}`}
                fill="none"
                stroke="#9a6fd1"
                strokeWidth="3"
                strokeDasharray="8 6"
              />
              <text x={frame.x + 16} y={frame.y + 26} fontSize="12" fontWeight="700" fill="#68429b">
                ReAct 循环 · 第 {frame.step} 轮
              </text>
              <text x={frame.x + 16} y={frame.y + 44} fontSize="11" fill="#765c9d">
                Context → Think → Act → Observation
              </text>
            </g>
          ))}
          {flow.edges.map((edge) => {
            const from = positions[edge.from];
            const to = positions[edge.to];
            if (!from || !to) return null;
            const source = flowNodeMap.get(edge.from);
            const target = flowNodeMap.get(edge.to);
            const isNextCycle = source?.kind === 'observation' && target?.kind === 'context';
            const middleX = (from.x + 168 + to.x) / 2;
            return (
              <g key={edge.from + '-' + edge.to}>
                <line
                  x1={from.x + 168}
                  y1={from.y + flowNodeHeight(source || { kind: 'start' } as FlowNode) / 2}
                  x2={to.x}
                  y2={to.y + flowNodeHeight(target || { kind: 'start' } as FlowNode) / 2}
                  stroke={isNextCycle ? '#765c9d' : '#9aa5b5'}
                  strokeWidth={isNextCycle ? '3' : '2'}
                  strokeDasharray={isNextCycle ? '7 5' : undefined}
                  markerEnd="url(#flow-arrow)"
                />
                {(isNextCycle || edge.label) && <text x={middleX} y={from.y + 38} textAnchor="middle" fontSize="11" fill={isNextCycle ? '#68429b' : '#526176'}>{isNextCycle ? '写回上下文，继续循环' : edge.label}</text>}
              </g>
            );
          })}
        </svg>
        {flow.nodes.map((node) => {
          const position = positions[node.id] || { x: 8, y: 8 };
          const colors = flowColors[node.kind];
          return (
            <div
              key={node.id}
              onPointerDown={(event) => startDrag(event, node)}
              onClick={() => onSelect(node.id)}
              style={{
                position: 'absolute',
                left: position.x,
                top: position.y,
                width: 168,
                minHeight: flowNodeHeight(node),
                boxSizing: 'border-box',
                padding: 11,
                border: `2px solid ${colors.border}`,
                borderRadius: 10,
                background: colors.background,
                cursor: 'grab',
                userSelect: 'none',
                zIndex: 1,
                boxShadow: selectedNodeId === node.id ? `0 0 0 3px ${colors.border}55` : undefined,
              }}
              title="拖动节点调整布局"
            >
              <div style={{ color: colors.label, fontWeight: 700, fontSize: 13 }}>{node.title}</div>
              <div style={{ marginTop: 8, color: '#4d586a', fontSize: 12, lineHeight: 1.35, wordBreak: 'break-word' }}>
                {node.subtitle}
              </div>
            </div>
          );
        })}
        {flow.nodes.length === 1 && (
          <div style={{ position: 'absolute', left: 24, top: 210, color: '#697386', fontSize: 13 }}>
            执行一步后，这里会出现紫色的 ReAct 循环框：Observation 的结果会写回下一轮 Context。
          </div>
        )}
      </div>
    </div>
  );
}

const nodeExplanations: Record<FlowNodeKind, string> = {
  start: '接收用户任务并创建一次运行。',
  context: '把系统规则、工具定义、用户输入和历史结果组装成下一轮模型输入。',
  brain: '调用模型，让模型选择下一步动作：调用工具、询问用户，或直接回答。',
  route: '当前 Agent 在作者连接允许的后继 Agent 中，选择下一位继续处理任务的 Agent。',
  tool: '按照模型给出的工具名和参数执行一次工具。',
  observation: '工具返回的结果。它会被写入上下文，供下一轮 Brain 参考。',
  wait: 'Agent 暂停执行，等待用户补充信息后再继续。',
  finish: 'Agent 已经决定结束循环，并给出最终答案。',
  error: '运行过程中发生错误；可以结合关联事件定位失败原因。',
};

function latestEvent(events: AgentEvent[], types: string[], step: number): AgentEvent | undefined {
  return [...events].reverse().find((event) => event.step === step && types.includes(event.type));
}

function nodeContent(node: FlowNode, events: AgentEvent[], run: RunResponse | null): unknown {
  if (node.kind === 'start') return latestEvent(events, ['run.start'], node.step)?.data ?? { input: run?.input };
  if (node.kind === 'context') return latestEvent(events, ['context.request', 'context.build'], node.step)?.data ?? {};
  if (node.kind === 'brain') return latestEvent(events, ['llm.end', 'llm.error'], node.step)?.data ?? {};
  if (node.kind === 'route') return latestEvent(events, ['agent.route'], node.step)?.data ?? {};
  if (node.kind === 'tool') return latestEvent(events, ['tool.start', 'tool.error'], node.step)?.data ?? {};
  if (node.kind === 'observation') {
    const observations = events
      .filter((event) => event.step === node.step && event.type === 'tool.end')
      .map((event) => event.data);
    return observations.length <= 1 ? (observations[0] ?? {}) : observations;
  }
  if (node.kind === 'wait') return latestEvent(events, ['agent.waiting_user'], node.step)?.data ?? {};
  if (node.kind === 'finish') return latestEvent(events, ['agent.finish'], node.step)?.data ?? {};
  return latestEvent(events, ['agent.error', 'llm.error', 'tool.error'], node.step)?.data ?? {};
}

function nodeRelatedEvents(node: FlowNode, events: AgentEvent[]): AgentEvent[] {
  const typesByKind: Record<FlowNodeKind, string[]> = {
    start: ['run.start'],
    context: ['context.build', 'context.request'],
    brain: ['llm.start', 'llm.end', 'action.validate', 'llm.error'],
    route: ['agent.route'],
    tool: ['tool.start', 'tool.error'],
    observation: ['tool.end'],
    wait: ['agent.waiting_user'],
    finish: ['agent.finish'],
    error: ['agent.error', 'llm.error', 'tool.error'],
  };
  return events.filter((event) => event.step === node.step && typesByKind[node.kind].includes(event.type));
}

function ContextSnapshot({ value }: { value: unknown }) {
  const snapshot = value && typeof value === 'object' ? value as Record<string, unknown> : {};
  const messages = Array.isArray(snapshot.messages) ? snapshot.messages as Array<Record<string, unknown>> : [];
  const tools = Array.isArray(snapshot.tools) ? snapshot.tools as Array<Record<string, unknown>> : [];
  const staticMessages = messages.filter((message) => message.role === 'system');
  const trajectory = messages.filter((message) => message.role !== 'system');
  const messageLine = (message: Record<string, unknown>) => {
    if (message.role === 'assistant' && Array.isArray(message.tool_calls)) return `assistant · tool_calls × ${message.tool_calls.length}`;
    if (message.role === 'tool') return `tool · ${String(message.tool_call_id || 'unlinked')}`;
    return `${String(message.role || 'message')} · ${stringValue(message.content, '')}`;
  };
  return (
    <div style={{ display: 'grid', gap: 10 }}>
      <div style={{ padding: 10, borderRadius: 8, background: '#f5edff', color: '#68429b', fontSize: 12 }}>
        请求协议：{String(snapshot.protocol || 'runtime')} · 静态前缀 {staticMessages.length} 条 · 动态轨迹 {trajectory.length} 条 · 顶层 tools {tools.length} 个
      </div>
      <div>
        <strong style={{ fontSize: 13 }}>静态前缀：System Prompt</strong>
        <div style={{ display: 'grid', gap: 5, marginTop: 6 }}>
          {staticMessages.map((message, index) => <div key={index} style={{ padding: 8, background: '#eef5ff', borderRadius: 6, fontSize: 12 }}>{messageLine(message)}</div>)}
        </div>
      </div>
      <div>
        <strong style={{ fontSize: 13 }}>工具定义：顶层 tools 字段</strong>
        <div style={{ padding: 8, background: '#fff5e5', borderRadius: 6, fontSize: 12 }}>{tools.length ? tools.map((tool) => String((tool.function as Record<string, unknown> | undefined)?.name || tool.name || 'tool')).join(' · ') : 'JSON Action 兼容模式：工具定义由教学适配层传递'}</div>
      </div>
      <div>
        <strong style={{ fontSize: 13 }}>动态轨迹：User / Assistant / Tool / Status</strong>
        <div style={{ display: 'grid', gap: 5, marginTop: 6 }}>
          {trajectory.map((message, index) => <div key={index} style={{ padding: 8, background: '#f5f7fb', borderRadius: 6, fontSize: 12, whiteSpace: 'pre-wrap' }}>{messageLine(message)}</div>)}
        </div>
      </div>
    </div>
  );
}

function NodeInspector({
  events,
  run,
  selectedNodeId,
}: {
  events: AgentEvent[];
  run: RunResponse | null;
  selectedNodeId: string;
}) {
  const node = buildFlow(events, run).nodes.find((item) => item.id === selectedNodeId);
  if (!node) {
    return <p style={{ color: '#5b6475', marginBottom: 0 }}>点击执行图中的节点查看详情。</p>;
  }
  const colors = flowColors[node.kind];
  const relatedEvents = nodeRelatedEvents(node, events);
  return (
    <div style={{ display: 'grid', gap: 12 }}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
        <span style={{ color: colors.label, fontWeight: 700 }}>{node.title}</span>
        <span style={{ color: '#697386', fontSize: 13 }}>step {node.step}</span>
      </div>
      <div>
        <strong>这个节点做什么</strong>
        <p style={{ margin: '5px 0 0', color: '#4d586a' }}>{nodeExplanations[node.kind]}</p>
      </div>
      <div>
        <strong>当前内容</strong>
        {node.kind === 'context' ? <div style={{ marginTop: 7 }}><ContextSnapshot value={nodeContent(node, events, run)} /></div> : (
          <pre style={{ overflowX: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word', background: '#f5f7fb', padding: 10, marginBottom: 0 }}>
            {JSON.stringify(nodeContent(node, events, run), null, 2)}
          </pre>
        )}
      </div>
      {relatedEvents.length > 0 && (
        <details>
          <summary>关联事件（{relatedEvents.length}）</summary>
          <div style={{ display: 'grid', gap: 6, marginTop: 8 }}>
            {relatedEvents.map((event) => (
              <details key={event.event_id}>
                <summary>{event.type}</summary>
                <pre style={{ overflowX: 'auto', background: '#f5f7fb', padding: 10 }}>
                  {formatEventData(event)}
                </pre>
              </details>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

export default function App() {
  const [status, setStatus] = useState('connecting...');
  const [llmStatus, setLlmStatus] = useState<LLMStatusResponse | null>(null);
  const [localModels, setLocalModels] = useState<LocalModelProfile[]>([]);
  const [localDevices, setLocalDevices] = useState<LocalExecutionDevice[]>([]);
  const [modelMetrics, setModelMetrics] = useState<ModelToolMetricsResponse | null>(null);
  const [metricsBusy, setMetricsBusy] = useState(false);
  const [cases, setCases] = useState<CaseTemplate[]>([]);
  const [selectedCase, setSelectedCase] = useState('');
  const [draftCaseId, setDraftCaseId] = useState('my-case');
  const [draftTitle, setDraftTitle] = useState('');
  const [draftDescription, setDraftDescription] = useState('');
  const [draftObjective, setDraftObjective] = useState('');
  const [draftInput, setDraftInput] = useState('');
  const [draftBrainType, setDraftBrainType] = useState<BrainType>('test');
  const [input, setInput] = useState('计算 12 * 7');
  const [brainType, setBrainType] = useState<BrainType>('test');
  const [model, setModel] = useState('qwen-tiny');
  const [run, setRun] = useState<RunResponse | null>(null);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  // ``null`` means live mode.  A numeric value is an exclusive event cursor:
  // 0 renders only the Run start card, 1 renders the first persisted event,
  // and so on.  This makes replay deterministic even when multiple events
  // belong to the same ReAct step.
  const [replayCursor, setReplayCursor] = useState<number | null>(null);
  const [replayPlaying, setReplayPlaying] = useState(false);
  const [replaySpeed, setReplaySpeed] = useState(1);
  const [selectedNodeId, setSelectedNodeId] = useState('run-start');
  const [replayText, setReplayText] = useState('');
  const [resumeInput, setResumeInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [editorDemoMode, setEditorDemoMode] = useState(false);
  const [editorPendingPhase, setEditorPendingPhase] = useState<'context' | 'llm' | null>(null);
  const streamRef = useRef<EventSource | null>(null);

  useEffect(() => {
    getHealth()
      .then((result) => setStatus(result.service + ' ' + result.version + ': ' + result.status))
      .catch((reason: Error) => setStatus('error: ' + reason.message));
    getLLMStatus()
      .then(setLlmStatus)
      .catch((reason: Error) => setError(reason.message));
    getLocalModels()
      .then((result) => { setLocalModels(result.models); setLocalDevices(result.devices); })
      .catch((reason: Error) => setError(reason.message));
    getModelToolMetrics()
      .then(setModelMetrics)
      .catch((reason: Error) => setError(reason.message));
    getCases()
      .then(setCases)
      .catch((reason: Error) => setError(reason.message));
    return () => streamRef.current?.close();
  }, []);

  function updateRun(next: RunResponse) {
    setRun(next);
    setEvents(next.events);
    setReplayCursor(null);
    setReplayPlaying(false);
    if (terminalStatuses.has(next.status) || next.status === 'paused') {
      streamRef.current?.close();
      streamRef.current = null;
    }
    if (terminalStatuses.has(next.status)) void refreshModelMetrics();
  }

  function attachStream(runId: string) {
    streamRef.current?.close();
    streamRef.current = openRunEventStream(
      runId,
      (event) => {
        setReplayCursor(null);
        setEvents((current) =>
          current.some((item) => item.event_id === event.event_id)
            ? current
            : [...current, event],
        );
      },
      () => {
        streamRef.current = null;
      },
    );
  }

  async function switchLocalModel(modelId: string, device = 'cpu', contextSize?: number) {
    const result = await activateLocalModel(modelId, device, contextSize);
    setLocalModels(result.models);
    setLocalDevices(result.devices);
    setLlmStatus(await getLLMStatus());
  }

  async function refreshModelMetrics() {
    try {
      setModelMetrics(await getModelToolMetrics());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function clearMetrics() {
    setMetricsBusy(true);
    try {
      setModelMetrics(await clearModelToolMetrics());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setMetricsBusy(false);
    }
  }

  function selectCase(caseId: string) {
    setSelectedCase(caseId);
    const template = cases.find((item) => item.case_id === caseId);
    if (!template) return;
    setInput(template.input);
    setBrainType(template.brain_type);
    setModel(template.brain_model || 'qwen-tiny');
  }

  async function saveCaseTemplate() {
    setBusy(true);
    setError('');
    try {
      const created = await createCaseTemplate({
        case_id: draftCaseId.trim(),
        title: draftTitle.trim(),
        description: draftDescription.trim(),
        objective: draftObjective.trim(),
        input: draftInput.trim(),
        brain_type: draftBrainType,
        brain_model: draftBrainType === 'tiny_llm' ? model : null,
        tools: draftBrainType === 'clarification'
          ? ['final_answer']
          : ['calculator', 'final_answer'],
      });
      setCases((current) => [...current, created]);
      setSelectedCase(created.case_id);
      setInput(created.input);
      setBrainType(created.brain_type);
      setModel(created.brain_model || 'qwen-tiny');
      setDraftCaseId('my-case');
      setDraftTitle('');
      setDraftDescription('');
      setDraftObjective('');
      setDraftInput('');
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  async function startRun() {
    setBusy(true);
    setError('');
    setReplayText('');
    setResumeInput('');
    setReplayCursor(null);
    setReplayPlaying(false);
    try {
      let created: RunResponse;
      if (selectedCase) {
        created = (await createCaseRun(selectedCase, input.trim())).run;
      } else {
        const agent = await createAgent(brainType, model);
        created = await createRun(agent.agent_id, input.trim());
      }
      updateRun(created);
      setSelectedNodeId('run-start');
      attachStream(created.run_id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  async function runWorkflow(workflow: WorkflowDocument) {
    setBusy(true);
    setEditorDemoMode(true);
    setError('');
    setReplayText('');
    setResumeInput('');
    setReplayCursor(null);
    setReplayPlaying(false);
    setEditorPendingPhase(null);
    try {
      const created = await createWorkflowRun(workflow);
      streamRef.current?.close();
      streamRef.current = null;
      setRun(created.run);
      setEvents([]);
      setSelectedNodeId('run-start');
      const displayed = new Set<string>();
      let current = created.run;
      while (current.status === 'created' || current.status === 'running') {
        // A step endpoint returns only after the model has finished. Keep the
        // LLM visibly active during that real wait, rather than making Context
        // appear slow merely because its event is replayed first afterwards.
        setEditorPendingPhase('context');
        const inferTimer = window.setTimeout(() => setEditorPendingPhase('llm'), 80);
        const next = await stepRun(current.run_id);
        window.clearTimeout(inferTimer);
        setEditorPendingPhase(null);
        const newEvents = next.events.filter((event) => !displayed.has(event.event_id));
        for (const event of newEvents) {
          displayed.add(event.event_id);
          setEvents((shown) => [...shown, event]);
          if (event.type === 'context.build') setSelectedNodeId(`context-${event.step}`);
          if (event.type === 'llm.start' || event.type === 'llm.end') setSelectedNodeId(`brain-${event.step}`);
          if (event.type === 'tool.start') setSelectedNodeId(`tool-${event.step}`);
          if (event.type === 'tool.end') setSelectedNodeId(`observation-${event.step}`);
          if (event.type === 'agent.finish') setSelectedNodeId(`finish-${event.step}`);
          await new Promise<void>((resolve) => window.setTimeout(resolve, 120));
        }
        current = next;
        setRun(next);
      }
      updateRun(current);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setEditorPendingPhase(null);
      setBusy(false);
    }
  }

  async function advance(action: 'step' | 'execute') {
    if (!run) return;
    setBusy(true);
    setError('');
    try {
      const next = action === 'step' ? await stepRun(run.run_id) : await executeRun(run.run_id);
      updateRun(next);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  async function pause() {
    if (!run) return;
    setBusy(true);
    setError('');
    try {
      updateRun(await pauseRun(run.run_id));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  async function resume() {
    if (!run) return;
    setBusy(true);
    setError('');
    try {
      const next = await resumeRun(
        run.run_id,
        run.status === 'waiting_user' ? resumeInput.trim() : undefined,
      );
      updateRun(next);
      setResumeInput('');
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  async function resumeWorkflowFromCanvas(input: string, selectionId?: string) {
    if (!run || run.status !== 'waiting_user') return;
    setBusy(true);
    setError('');
    try {
      updateRun(await resumeRun(run.run_id, input.trim(), selectionId));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
      throw reason;
    } finally {
      setBusy(false);
    }
  }

  async function replay() {
    if (!run) return;
    setBusy(true);
    setError('');
    try {
      const result = await getReplay(run.run_id);
      setEvents(result.events);
      setReplayText(JSON.stringify(result, null, 2));
      setReplayCursor(0);
      setReplayPlaying(false);
      setSelectedNodeId('run-start');
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  const canPause = run && (run.status === 'created' || run.status === 'running');
  const canExecute = run && (run.status === 'created' || run.status === 'running');
  const canStep = run && (run.status === 'created' || run.status === 'running');
  const canResume = run && (run.status === 'paused' || run.status === 'waiting_user');
  const displayedEvents = replayCursor === null ? events : events.slice(0, replayCursor);
  const replayEvent = replayCursor && replayCursor > 0 ? events[replayCursor - 1] : null;
  const replayPositionLabel = replayEvent
    ? `事件 ${replayCursor} / ${events.length} · step ${replayEvent.step} · ${replayEvent.type}`
    : `事件 ${replayCursor ?? 0} / ${events.length} · Run start`;

  function moveReplayCursor(nextCursor: number, stopPlayback = true) {
    const cursor = Math.max(0, Math.min(events.length, nextCursor));
    setReplayCursor(cursor);
    if (stopPlayback) setReplayPlaying(false);
    if (cursor === 0) {
      setSelectedNodeId('run-start');
      return;
    }
    const visible = events.slice(0, cursor);
    const flow = buildFlow(visible, run);
    setSelectedNodeId(flow.nodes.at(-1)?.id || 'run-start');
  }

  useEffect(() => {
    if (!replayPlaying || replayCursor === null) return undefined;
    if (replayCursor >= events.length) {
      setReplayPlaying(false);
      return undefined;
    }
    const timer = window.setTimeout(
      () => moveReplayCursor(replayCursor + 1, false),
      Math.round(800 / replaySpeed),
    );
    return () => window.clearTimeout(timer);
  }, [events.length, replayCursor, replayPlaying, replaySpeed, run]);

  return (
    <main
      style={{
        fontFamily: 'system-ui, sans-serif',
        minHeight: '100vh',
        background: '#f5f7fb',
        color: '#172033',
        padding: 0,
      }}
    >
      <AgentEditor
        navigationPanels={
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(230px, 0.3fr) minmax(0, 1fr)', gap: 12 }}>
            <section style={{ ...panelStyle, padding: 12 }}>
              <strong>服务状态</strong>
              <p style={{ margin: '7px 0 4px', fontSize: 12 }}>{status}</p>
              <p style={{ margin: 0, color: llmStatus?.available ? '#18794e' : '#a33a2b', fontSize: 12 }}>
                本地 LLM：{llmStatus ? `${llmStatus.available ? 'available' : 'unavailable'} · ${llmStatus.model}${llmStatus.context_size ? ` · Context ${llmStatus.context_size}` : ''}` : 'checking...'}
              </p>
            </section>
            <section style={{ ...panelStyle, padding: 12, minWidth: 0 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
                <div>
                  <strong>模型工具调用对比</strong>
                  <span style={{ color: '#697386', marginLeft: 8, fontSize: 11 }}>计划 → 校验 → 执行 → 成功</span>
                </div>
                <button onClick={() => void clearMetrics()} disabled={metricsBusy}>{metricsBusy ? '正在清空…' : '清空统计'}</button>
              </div>
              {!modelMetrics || modelMetrics.models.length === 0 ? (
                <p style={{ color: '#697386', margin: '8px 0 0', fontSize: 12 }}>尚无本统计窗口内的模型工具调用。</p>
              ) : (
                <div style={{ overflowX: 'auto', marginTop: 8 }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                    <thead><tr style={{ textAlign: 'left', borderBottom: '1px solid #d9dee8' }}><th style={{ padding: 6 }}>模型</th><th>运行</th><th>调用漏斗</th><th>计划→校验</th><th>计划→成功</th><th>执行成功</th><th>决策错误分类</th><th>工具失败 / 拦截</th></tr></thead>
                    <tbody>{modelMetrics.models.map((metric) => (
                      <tr key={`${metric.provider}-${metric.model}`} style={{ borderBottom: '1px solid #edf0f5' }}>
                        <td style={{ padding: 6 }}><strong>{metric.model}</strong><br /><span style={{ color: '#697386' }}>{metric.provider}</span></td>
                        <td>{metric.completed_runs} / {metric.runs}</td>
                        <td>{metric.tool_calls_requested} → {metric.tool_calls_validated} → {metric.tool_calls_started} → {metric.tool_calls_succeeded}</td>
                        <td>{percentage(metric.proposal_validation_rate)}</td><td>{percentage(metric.plan_to_tool_success_rate)}</td><td>{percentage(metric.tool_execution_success_rate)}</td>
                        <td>共 {metric.decision_errors} · {decisionBreakdownText(metric.decision_error_breakdown)}</td>
                        <td>工具 {metric.tool_calls_failed} · 拦截 {metric.tool_calls_blocked}</td>
                      </tr>
                    ))}</tbody>
                  </table>
                </div>
              )}
            </section>
          </div>
        }
        onRun={runWorkflow}
        disabled={busy}
        demoMode={editorDemoMode}
        demoEvents={events}
        demoRunStatus={run?.status === 'created' && events.length > 0 ? 'running' : run?.status}
        demoRunId={run?.run_id}
        demoError={run?.error}
        demoPendingPhase={editorPendingPhase}
        demoOutput={run?.output}
        waitingQuestion={run?.status === 'waiting_user' ? run.waiting_question : null}
        waitingChoices={run?.status === 'waiting_user' ? run.waiting_choices ?? [] : []}
        waitingResponseSchema={run?.status === 'waiting_user' ? run.waiting_response_schema ?? null : null}
        onSubmitUserInput={resumeWorkflowFromCanvas}
        onExitDemo={() => setEditorDemoMode(false)}
        localModels={localModels}
        localDevices={localDevices}
        onActivateLocalModel={switchLocalModel}
      />

      {false && run && <>
      <div style={{ height: 16 }} />

      <section style={{ ...panelStyle, marginBottom: 16 }}>
        <h2 style={{ fontSize: 18, marginTop: 0 }}>教学案例与运行控制</h2>
        <label style={{ display: 'block', marginBottom: 10 }}>
          案例模板
          <select
            value={selectedCase}
            onChange={(event) => selectCase(event.target.value)}
            style={{ display: 'block', width: '100%', marginTop: 5, padding: 9 }}
          >
            <option value="">自定义运行</option>
            {cases.map((template) => (
              <option key={template.case_id} value={template.case_id}>
                {template.title}
              </option>
            ))}
          </select>
        </label>
        <details style={{ marginBottom: 12 }}>
          <summary>创建自定义教学案例模板</summary>
          <div style={{ display: 'grid', gap: 8, marginTop: 10 }}>
            <input
              value={draftCaseId}
              onChange={(event) => setDraftCaseId(event.target.value)}
              placeholder="案例 ID，例如 my-case"
            />
            <input
              value={draftTitle}
              onChange={(event) => setDraftTitle(event.target.value)}
              placeholder="案例标题"
            />
            <input
              value={draftDescription}
              onChange={(event) => setDraftDescription(event.target.value)}
              placeholder="案例说明"
            />
            <input
              value={draftObjective}
              onChange={(event) => setDraftObjective(event.target.value)}
              placeholder="教学目标"
            />
            <input
              value={draftInput}
              onChange={(event) => setDraftInput(event.target.value)}
              placeholder="默认任务输入"
            />
            <select
              value={draftBrainType}
              onChange={(event) => setDraftBrainType(event.target.value as BrainType)}
            >
              <option value="test">test（演示）</option>
              <option value="clarification">ask_user（追问）</option>
              <option value="tiny_llm">tiny_llm（本地模型）</option>
            </select>
            <button
              onClick={() => void saveCaseTemplate()}
              disabled={
                busy ||
                !draftCaseId.trim() ||
                !draftTitle.trim() ||
                !draftDescription.trim() ||
                !draftObjective.trim() ||
                !draftInput.trim()
              }
            >
              保存案例模板
            </button>
          </div>
        </details>
        {selectedCase && (
          <p style={{ color: '#5b6475', marginTop: 0 }}>
            {cases.find((item) => item.case_id === selectedCase)?.objective}
          </p>
        )}
        <label style={{ display: 'block', marginBottom: 10 }}>
          任务
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            style={{ display: 'block', width: '100%', marginTop: 5, padding: 9, boxSizing: 'border-box' }}
          />
        </label>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'end' }}>
          <label>
            Brain
            <select
              value={brainType}
              onChange={(event) => setBrainType(event.target.value as BrainType)}
              style={{ display: 'block', marginTop: 5, padding: 8 }}
            >
              <option value="test">test（演示）</option>
              <option value="clarification">ask_user（追问演示）</option>
              <option value="tiny_llm">tiny_llm（真实本地模型）</option>
            </select>
          </label>
          <label style={{ flex: '1 1 220px' }}>
            Model
            <input
              value={model}
              onChange={(event) => setModel(event.target.value)}
              style={{ display: 'block', width: '100%', marginTop: 5, padding: 8, boxSizing: 'border-box' }}
            />
          </label>
          <button onClick={() => void startRun()} disabled={busy || !input.trim()}>
            新建运行
          </button>
          <button onClick={() => void advance('step')} disabled={busy || !canStep}>
            执行一步
          </button>
          <button onClick={() => void advance('execute')} disabled={busy || !canExecute}>
            运行完成
          </button>
          <button onClick={() => void pause()} disabled={busy || !canPause}>
            暂停
          </button>
          <button onClick={() => void resume()} disabled={busy || !canResume || run?.status === 'waiting_user'}>
            继续
          </button>
          <button onClick={() => void replay()} disabled={busy || !run}>
            回放
          </button>
        </div>
        {run?.status === 'waiting_user' && (
          <div style={{ marginTop: 14, padding: 12, background: '#fff8e6', borderRadius: 8 }}>
            <strong>{Array.isArray(run?.memory.pending_tool_calls) ? '工具调用等待确认' : 'Agent 需要你的补充'}</strong>
            <p>{run?.waiting_question}</p>
            <input
              value={resumeInput}
              onChange={(event) => setResumeInput(event.target.value)}
              placeholder={Array.isArray(run?.memory.pending_tool_calls) ? '输入“确认”或“拒绝”' : '输入补充信息'}
              style={{ width: '100%', padding: 9, boxSizing: 'border-box' }}
            />
            <button
              onClick={() => void resume()}
              disabled={busy || !resumeInput.trim()}
              style={{ marginTop: 8 }}
            >
              {Array.isArray(run?.memory.pending_tool_calls) ? '确认结果并继续' : '提交并继续'}
            </button>
          </div>
        )}
        {run && (
          <p style={{ marginBottom: 0 }}>
            <strong>{run?.status}</strong> · step {run?.step} · run {run?.run_id}
            {run?.output ? ' · output: ' + run?.output : ''}
          </p>
        )}
        {error && <p style={{ color: '#a33a2b', marginBottom: 0 }}>错误：{error}</p>}
      </section>

      <section style={{ ...panelStyle, marginBottom: 16 }}>
        <h2 style={{ fontSize: 18, marginTop: 0 }}>ReAct 执行图</h2>
        <p style={{ color: '#5b6475', marginTop: 0 }}>
          节点按真实事件生成；拖动节点可以调整布局，点击节点查看详细说明和当前内容。
        </p>
        {events.length > 0 && (
          <div style={{ border: '1px solid #d9dee8', borderRadius: 9, padding: 12, marginBottom: 14, background: '#f8fbff' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
              <strong>事件回放控制</strong>
              <span style={{ color: '#4d586a', fontSize: 13 }}>
                {replayCursor === null ? `实时跟随 · ${events.length} 条事件` : replayPositionLabel}
              </span>
            </div>
            <input
              aria-label="回放事件游标"
              type="range"
              min="0"
              max={events.length}
              value={replayCursor ?? events.length}
              onChange={(event) => moveReplayCursor(Number(event.target.value))}
              style={{ width: '100%', margin: '10px 0' }}
            />
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
              <button onClick={() => moveReplayCursor((replayCursor ?? events.length) - 1)} disabled={(replayCursor ?? events.length) === 0}>上一步</button>
              <button onClick={() => {
                if (replayCursor === null || replayCursor >= events.length) moveReplayCursor(0, false);
                setReplayPlaying((playing) => !playing);
              }}>
                {replayPlaying ? '暂停播放' : '自动播放'}
              </button>
              <button onClick={() => moveReplayCursor((replayCursor ?? events.length) + 1)} disabled={(replayCursor ?? events.length) >= events.length}>下一步</button>
              <button onClick={() => { setReplayCursor(null); setReplayPlaying(false); }} disabled={replayCursor === null}>回到实时末尾</button>
              <label style={{ fontSize: 13, color: '#4d586a' }}>
                速度
                <select value={replaySpeed} onChange={(event) => setReplaySpeed(Number(event.target.value))} style={{ marginLeft: 6, padding: 5 }}>
                  <option value={0.5}>0.5×</option>
                  <option value={1}>1×</option>
                  <option value={2}>2×</option>
                  <option value={4}>4×</option>
                </select>
              </label>
            </div>
          </div>
        )}
        <FlowCanvas
          events={displayedEvents}
          run={run}
          selectedNodeId={selectedNodeId}
          onSelect={setSelectedNodeId}
        />
      </section>

      <section style={{ ...panelStyle, marginBottom: 16 }}>
        <h2 style={{ fontSize: 18, marginTop: 0 }}>节点详情</h2>
        <NodeInspector events={displayedEvents} run={run} selectedNodeId={selectedNodeId} />
      </section>

      <section style={panelStyle}>
        <h2 style={{ fontSize: 18, marginTop: 0 }}>Event Timeline（SSE / Replay）</h2>
        {displayedEvents.length === 0 ? (
          <p style={{ color: '#5b6475' }}>创建运行后，事件会按真实执行顺序出现在这里。</p>
        ) : (
          <div style={{ display: 'grid', gap: 8 }}>
            {displayedEvents.map((event) => (
              <details
                key={event.event_id}
                open={
                  event.type === 'agent.finish' ||
                  event.type === 'agent.error' ||
                  event.type === 'agent.waiting_user'
                }
              >
                <summary>
                  step {event.step} · {event.type} · {event.node}
                </summary>
                <pre style={{ overflowX: 'auto', background: '#f5f7fb', padding: 10 }}>
                  {formatEventData(event)}
                </pre>
              </details>
            ))}
          </div>
        )}
        {replayText && (
          <details style={{ marginTop: 16 }}>
            <summary>Replay JSON</summary>
            <pre style={{ overflowX: 'auto', background: '#f5f7fb', padding: 10 }}>{replayText}</pre>
          </details>
        )}
      </section>
      </>}
    </main>
  );
}
