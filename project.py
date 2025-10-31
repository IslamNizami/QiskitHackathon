import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import hashlib
import secrets
from collections import Counter
import random
def _pack_bits_to_hex(bits):
        if not bits:
            return ""
        bit_string = ''.join(map(str, bits))
        pad = (8 - len(bit_string) % 8) % 8
        bit_string += '0' * pad
        key_int = int(bit_string, 2)
        byte_len = len(bit_string) // 8
        return key_int.to_bytes(byte_len, 'big').hex()

def _toeplitz_hash(bits_in, m_out, rng=None):
       
    import random
    rng = rng or random.Random()
    n = len(bits_in)

    if not hasattr(rng, "randbelow"):
        def rb(): return rng.getrandbits(1)
    else:
        def rb(): return rng.randbelow(2)

        seed = [rb() for _ in range(n + m_out - 1)]


        # Convert to numpy uint8 for fast mod-2 operations
        x = _np.array(bits_in, dtype=_np.uint8)
        out = _np.zeros(m_out, dtype=_np.uint8)

        # Each output bit j is dot( row_j , x ) mod 2,
        # where row_j[k] = seed[j + (n-1 - k)]  with Toeplitz structure.
        # We can compute via sliding window XORs.
        for j in range(m_out):
            # Build row implicitly: row_j[k] = seed[j + n - 1 - k]
            # Equivalent to reversing x and taking a length-n window of seed
            # starting at index j and XOR-dot with x.
            s_slice = seed[j : j + n]                 # length n
            # XOR-dot over GF(2) is sum( x & s_slice ) mod 2
            out[j] = ( (x & _np.array(s_slice, dtype=_np.uint8)).sum() ) & 1

        return out.tolist()

class BB84Protocol:
    def __init__(self, key_length=256, error_threshold=0.11):
        self.key_length = key_length
        self.error_threshold = error_threshold
        self.backend = AerSimulator()
        
    def alice_prepare_qubits(self):
        """Alice generates random bits and random bases for each qubit"""
        alice_bits = [secrets.randbelow(2) for _ in range(self.key_length)]
        alice_bases = [secrets.choice(['Z', 'X']) for _ in range(self.key_length)]
        
        # Create quantum circuits for each qubit
        circuits = []
        for bit, basis in zip(alice_bits, alice_bases):
            qc = QuantumCircuit(1, 1)
            
            if basis == 'Z':
                # Computational basis
                if bit == 1:
                    qc.x(0)  # |1⟩ state
                # else: |0⟩ state (default)
            else:  # X basis (Hadamard basis)
                if bit == 0:
                    qc.h(0)  # |+⟩ state
                else:
                    qc.x(0)
                    qc.h(0)  # |-⟩ state
            
            circuits.append(qc)
        
        return alice_bits, alice_bases, circuits
    
    def bob_measure_qubits(self, circuits, bob_bases=None):
        """Bob measures qubits in randomly chosen bases"""
        if bob_bases is None:
            bob_bases = [secrets.choice(['Z', 'X']) for _ in range(self.key_length)]
        
        bob_results = []
        measured_circuits = []
        
        for qc, basis in zip(circuits, bob_bases):
            measured_qc = qc.copy()
            measured_qc.barrier()
            
            if basis == 'X':
                measured_qc.h(0)  # Change to X basis for measurement
            
            measured_qc.measure(0, 0)
            measured_circuits.append(measured_qc)
            
            # Execute measurement
            job = self.backend.run(measured_qc, shots=1, memory=True)
            result = job.result()
            measurement = int(result.get_memory()[0])
            bob_results.append(measurement)
        
        return bob_results, bob_bases, measured_circuits
    
    def sift_keys(self, alice_bits, alice_bases, bob_results, bob_bases):
        """Keep only bits where Alice and Bob used the same basis"""
        sifted_alice_key = []
        sifted_bob_key = []
        matching_indices = []
        
        for i, (a_base, b_base) in enumerate(zip(alice_bases, bob_bases)):
            if a_base == b_base:
                sifted_alice_key.append(alice_bits[i])
                sifted_bob_key.append(bob_results[i])
                matching_indices.append(i)
        
        return sifted_alice_key, sifted_bob_key, matching_indices
    
    def estimate_error_rate(self, alice_key, bob_key, test_fraction=0.5):
        """Estimate quantum bit error rate (QBER)"""
        if len(alice_key) == 0:
            return 1.0, []  # Maximum error if no key
        
        # Use a portion of the key for testing
        test_size = max(1, int(len(alice_key) * test_fraction))
        test_indices = secrets.SystemRandom().sample(range(len(alice_key)), test_size)
        
        errors = 0
        for idx in test_indices:
            if alice_key[idx] != bob_key[idx]:
                errors += 1
        
        qber = errors / len(test_indices)
        return qber, test_indices
    

    def perform_privacy_amplification(self, raw_key, test_indices):
  
        # 1) Remove test bits
        final_bits = [bit for i, bit in enumerate(raw_key) if i not in test_indices]

        # Decide output length m_out (shrinkage). Simple choice: half the remaining bits.
        # You can tune this (e.g., m_out = int(len(final_bits) * (1 - safety_margin)) ).
        m_out = max(1, len(final_bits) // 2)

        # 2) Toeplitz universal hash
        hashed_bits = _toeplitz_hash(final_bits, m_out, rng=secrets.SystemRandom())

        # 3) Pack to hex for display/storage
        key_hex = _pack_bits_to_hex(hashed_bits)

        return key_hex, final_bits


    
    def simulate_eavesdropping(self, circuits, eavesdrop_fraction=0.3):
        """Simulate Eve intercepting and measuring some qubits"""
        eve_bases = [secrets.choice(['Z', 'X']) for _ in range(len(circuits))]
        intercepted_circuits = []
        
        for i, qc in enumerate(circuits):
            if random.random() < eavesdrop_fraction:
                # Eve intercepts and measures
                eve_qc = qc.copy()
                
                if eve_bases[i] == 'X':
                    eve_qc.h(0)
                
                eve_qc.measure(0, 0)
                job = self.backend.run(eve_qc, shots=1, memory=True)
                result = job.result()
                eve_measurement = int(result.get_memory()[0])
                
                # Eve prepares new qubit based on her measurement
                new_qc = QuantumCircuit(1, 1)
                if eve_bases[i] == 'Z':
                    if eve_measurement == 1:
                        new_qc.x(0)
                else:  # X basis
                    if eve_measurement == 0:
                        new_qc.h(0)  # |+⟩
                    else:
                        new_qc.x(0)
                        new_qc.h(0)  # |-⟩
                
                intercepted_circuits.append(new_qc)
            else:
                intercepted_circuits.append(qc.copy())
        
        return intercepted_circuits

def run_bb84_simulation(num_simulations=10, key_length=100):
    """Run multiple BB84 simulations and collect statistics"""
    bb84 = BB84Protocol(key_length=key_length)
    
    results = {
        'no_eve': {'qber': [], 'key_rate': [], 'success_rate': 0},
        'with_eve': {'qber': [], 'key_rate': [], 'success_rate': 0}
    }
    
    print("Running BB84 Simulations...")
    print("=" * 50)
    
    # Simulation without Eve
    print("\n1. Simulation WITHOUT Eavesdropper:")
    print("-" * 40)
    
    successful_runs = 0
    for i in range(num_simulations):
        # Alice prepares qubits
        alice_bits, alice_bases, circuits = bb84.alice_prepare_qubits()
        
        # Bob measures
        bob_results, bob_bases, _ = bb84.bob_measure_qubits(circuits)
        
        # Sift keys
        sifted_alice, sifted_bob, _ = bb84.sift_keys(alice_bits, alice_bases, bob_results, bob_bases)
        
        if len(sifted_alice) > 0:
            # Estimate error rate
            qber, test_indices = bb84.estimate_error_rate(sifted_alice, sifted_bob)
            
            # Privacy amplification
            final_key, raw_key = bb84.perform_privacy_amplification(sifted_alice, test_indices)
            
            results['no_eve']['qber'].append(qber)
            results['no_eve']['key_rate'].append(len(raw_key) / key_length)
            
            if qber < bb84.error_threshold:
                successful_runs += 1
                print(f"  Run {i+1}: QBER = {qber:.3f}, Key Rate = {len(raw_key)/key_length:.3f} ✓")
            else:
                print(f"  Run {i+1}: QBER = {qber:.3f}, Key Rate = {len(raw_key)/key_length:.3f} ✗")
        else:
            print(f"  Run {i+1}: No matching bases - protocol failed")
    
    results['no_eve']['success_rate'] = successful_runs / num_simulations
    
    # Simulation with Eve
    print("\n2. Simulation WITH Eavesdropper (30% interception):")
    print("-" * 55)
    
    successful_runs = 0
    for i in range(num_simulations):
        # Alice prepares qubits
        alice_bits, alice_bases, circuits = bb84.alice_prepare_qubits()
        
        # Eve intercepts some qubits
        intercepted_circuits = bb84.simulate_eavesdropping(circuits, eavesdrop_fraction=0.3)
        
        # Bob measures intercepted qubits
        bob_results, bob_bases, _ = bb84.bob_measure_qubits(intercepted_circuits)
        
        # Sift keys
        sifted_alice, sifted_bob, _ = bb84.sift_keys(alice_bits, alice_bases, bob_results, bob_bases)
        
        if len(sifted_alice) > 0:
            # Estimate error rate
            qber, test_indices = bb84.estimate_error_rate(sifted_alice, sifted_bob)
            
            results['with_eve']['qber'].append(qber)
            results['with_eve']['key_rate'].append(len(sifted_alice) / key_length)
            
            if qber < bb84.error_threshold:
                successful_runs += 1
                print(f"  Run {i+1}: QBER = {qber:.3f}, Key Rate = {len(sifted_alice)/key_length:.3f} ✗ (Eve undetected!)")
            else:
                print(f"  Run {i+1}: QBER = {qber:.3f}, Key Rate = {len(sifted_alice)/key_length:.3f} ✗ (Eve detected!)")
        else:
            print(f"  Run {i+1}: No matching bases - protocol failed")
    
    results['with_eve']['success_rate'] = successful_runs / num_simulations
    
    return results

def plot_results(results):
    """Plot comparison of BB84 performance with and without eavesdropping"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # QBER comparison
    ax1.boxplot([results['no_eve']['qber'], results['with_eve']['qber']], 
                labels=['No Eve', 'With Eve'])
    ax1.set_ylabel('Quantum Bit Error Rate (QBER)')
    ax1.set_title('QBER Distribution')
    ax1.axhline(y=0.11, color='r', linestyle='--', label='Security Threshold')
    ax1.legend()
    
    # Key rate comparison
    ax2.boxplot([results['no_eve']['key_rate'], results['with_eve']['key_rate']], 
                labels=['No Eve', 'With Eve'])
    ax2.set_ylabel('Final Key Rate (bits/raw bit)')
    ax2.set_title('Key Rate Efficiency')
    
    # Success rate
    scenarios = ['No Eve', 'With Eve']
    success_rates = [results['no_eve']['success_rate'], results['with_eve']['success_rate']]
    bars = ax3.bar(scenarios, success_rates, color=['green', 'red'])
    ax3.set_ylabel('Success Rate')
    ax3.set_title('Protocol Success Rate')
    ax3.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom')
    
    # Theoretical QBER explanation
    theoretical_qber = [0.0, 0.25]  # No Eve vs With Eve (theoretical)
    ax4.bar(['Theoretical\nNo Eve', 'Theoretical\nWith Eve'], theoretical_qber, 
            color=['lightblue', 'lightcoral'])
    ax4.set_ylabel('Theoretical QBER')
    ax4.set_title('Theoretical QBER Values')
    
    plt.tight_layout()
    plt.show()

def demonstrate_single_bb84_run():
    """Demonstrate a single BB84 protocol run step by step"""
    print("\n" + "="*60)
    print("SINGLE BB84 PROTOCOL DEMONSTRATION")
    print("="*60)
    
    bb84 = BB84Protocol(key_length=10)  # Small key for demonstration
    
    print("\nStep 1: Alice prepares qubits")
    print("-" * 35)
    alice_bits, alice_bases, circuits = bb84.alice_prepare_qubits()
    print(f"Alice's random bits:  {alice_bits}")
    print(f"Alice's random bases: {alice_bases}")
    
    print("\nStep 2: Bob measures qubits")
    print("-" * 30)
    bob_results, bob_bases, _ = bb84.bob_measure_qubits(circuits)
    print(f"Bob's random bases:   {bob_bases}")
    print(f"Bob's measurements:   {bob_results}")
    
    print("\nStep 3: Basis sifting")
    print("-" * 20)
    sifted_alice, sifted_bob, matching_indices = bb84.sift_keys(alice_bits, alice_bases, bob_results, bob_bases)
    print(f"Matching bases at indices: {matching_indices}")
    print(f"Sifted Alice key: {sifted_alice}")
    print(f"Sifted Bob key:   {sifted_bob}")
    
    print("\nStep 4: Error estimation")
    print("-" * 25)
    qber, test_indices = bb84.estimate_error_rate(sifted_alice, sifted_bob, test_fraction=0.5)
    print(f"Test bits at indices: {test_indices}")
    print(f"Estimated QBER: {qber:.3f}")
    
    if qber < bb84.error_threshold:
        print("✓ QBER below threshold - Channel is secure!")
        final_key, raw_key = bb84.perform_privacy_amplification(sifted_alice, test_indices)
        print(f"Final shared key: {final_key[:16]}...")
    else:
        print("✗ QBER above threshold - Possible eavesdropping detected!")

def compare_with_other_qkd_protocols():
    """Compare BB84 with other QKD protocols"""
    print("\n" + "="*60)
    print("COMPARISON WITH OTHER QKD PROTOCOLS")
    print("="*60)
    
    protocols = {
        'BB84': {
            'Year': 1984,
            'Security': 'Unconditional',
            'Qubits': 'Single photons',
            'Efficiency': '~50%',
            'Implementation': 'Commercial',
            'Vulnerabilities': 'Photon number splitting'
        },
        'E91 (Ekert)': {
            'Year': 1991,
            'Security': 'Entanglement-based',
            'Qubits': 'Entangled pairs',
            'Efficiency': '~50%',
            'Implementation': 'Experimental',
            'Vulnerabilities': 'Entanglement quality'
        },
        'B92': {
            'Year': 1992,
            'Security': 'Unconditional',
            'Qubits': 'Two non-orthogonal states',
            'Efficiency': '~25%',
            'Implementation': 'Limited',
            'Vulnerabilities': 'Similar to BB84'
        },
        'COW (Coherent One-Way)': {
            'Year': 2003,
            'Security': 'Computational',
            'Qubits': 'Coherent states',
            'Efficiency': 'High',
            'Implementation': 'Commercial',
            'Vulnerabilities': 'Photon number splitting'
        }
    }
    
    print("\nProtocol Comparison Table:")
    print("-" * 90)
    print(f"{'Protocol':<10} {'Year':<6} {'Security':<15} {'Qubits':<20} {'Efficiency':<10} {'Status':<12} {'Vulnerabilities'}")
    print("-" * 90)
    for protocol, info in protocols.items():
        print(f"{protocol:<10} {info['Year']:<6} {info['Security']:<15} {info['Qubits']:<20} {info['Efficiency']:<10} {info['Implementation']:<12} {info['Vulnerabilities']}")

# Main execution
if __name__ == "__main__":
    try:
        # Single run demonstration
        demonstrate_single_bb84_run()
        
        # Multiple simulations for statistics
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS (20 simulations)")
        print("="*60)
        results = run_bb84_simulation(num_simulations=20, key_length=100)
        
        # Plot results
        plot_results(results)
        
        # Print summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"No Eve Scenario:")
        print(f"  Average QBER: {np.mean(results['no_eve']['qber']):.3f}")
        print(f"  Average Key Rate: {np.mean(results['no_eve']['key_rate']):.3f}")
        print(f"  Success Rate: {results['no_eve']['success_rate']:.1%}")
        
        print(f"\nWith Eve Scenario:")
        print(f"  Average QBER: {np.mean(results['with_eve']['qber']):.3f}")
        print(f"  Average Key Rate: {np.mean(results['with_eve']['key_rate']):.3f}")
        print(f"  Success Rate: {results['with_eve']['success_rate']:.1%}")
        
        # Protocol comparison
        compare_with_other_qkd_protocols()
        
        print("\n" + "="*60)
        print("EDUCATIONAL INSIGHTS")
        print("="*60)
        print("""
Key Takeaways:
1. BB84 provides information-theoretic security based on quantum principles
2. Eavesdropping increases QBER from ~0% to ~25%, making detection straightforward
3. The protocol is robust but has practical limitations (distance, rate)
4. BB84 complements classical post-quantum cryptography for comprehensive security
        
Future Directions:
- Quantum repeaters for long-distance QKD
- Integrated photonic chips for cost reduction
- Hybrid QKD-classical systems for practical deployment
- Standardization and interoperability efforts
        """)
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("Make sure you have installed all required packages:")
        print("pip install qiskit qiskit-aer matplotlib numpy")