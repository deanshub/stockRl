export default class Environment{
  constructor(state) {
    this.state = state
  }

  reset() {
    throw new Error('"reset" not implemented in this environment')
  }
  render() {
    throw new Error('"Render" not implemented in this environment')
  }

  randomAction() {
    throw new Error('"randomAction" not implemented in this environment')
  }
  actionSpace() {
    throw new Error('"actionSpace" not implemented in this environment')
  }
  step() {
    throw new Error('"step" not implemented in this environment')
  }
}
