import serial # Or other library for your specific radio module (e.g., spidev for SPI)
import struct
import threading
import numpy as np
import traceback
# from scipy.interpolate import interp1d # Not directly used in your provided code
# from scipy.misc import derivative     # Not directly used in your provided code
import nmrglue as ng

# --- Radio Hardware Abstraction Layer (Conceptual) ---
# This part depends heavily on your chosen radio module.
# For example, using a simple serial interface for a basic radio modem.

class RadioInterface:
    def __init__(self, serial_port, baud_rate, node_id):
        self.ser = serial.Serial(serial_port, baud_rate, timeout=1)
        self.node_id = node_id
        self.receive_buffer = bytearray()
        self.rx_lock = threading.Lock() # For managing receive buffer
        print(f"Radio Interface initialized on {serial_port} with ID {node_id}")

    def send_packet(self, destination_id, packet_type, payload):
        # In a real radio system, you'd add:
        # 1. Start-of-packet delimiter
        # 2. Source ID, Destination ID
        # 3. Packet Type (your operation_code)
        # 4. Payload length
        # 5. Payload data
        # 6. Checksum/CRC for error detection
        # 7. End-of-packet delimiter

        # For simplicity, let's just prepend destination and type for now.
        # This is very basic and lacks error checking, retries, ACK, etc.
        packet = struct.pack('!BB', destination_id, packet_type) + payload
        try:
            # Implement a basic MAC-like mechanism if needed (e.g., listen before talk)
            # For simplicity, sending directly.
            self.ser.write(packet)
            # print(f"Sent {len(packet)} bytes to {destination_id} (Type {packet_type})")
            return True
        except serial.SerialException as e:
            print(f"Error sending data via radio: {e}")
            return False

    def receive_packet(self):
        # This is a blocking read, you'd typically run this in a separate thread.
        # It needs robust parsing for packet delimiters, checksums, and lengths.
        # For demonstration, a very basic non-robust read.
        try:
            raw_data = self.ser.read_all() # Read all available bytes
            if raw_data:
                with self.rx_lock:
                    self.receive_buffer.extend(raw_data)

                # Implement robust packet parsing here.
                # Look for delimiters, check checksums, extract fields.
                # For this example, let's assume packets are: DestID (1B), Type (1B), Length (4B), Payload (Variable)
                # This is highly simplified and will break with partial reads.
                if len(self.receive_buffer) >= 6: # Minimum for DestID, Type, Length
                    dest_id, pkt_type = struct.unpack_from('!BB', self.receive_buffer, 0)
                    payload_len = struct.unpack_from('!I', self.receive_buffer, 2)[0]

                    if len(self.receive_buffer) >= 6 + payload_len:
                        payload = self.receive_buffer[6 : 6 + payload_len]
                        # Remove processed data from buffer
                        self.receive_buffer = self.receive_buffer[6 + payload_len:]
                        return dest_id, pkt_type, payload
            return None, None, None # No complete packet yet
        except serial.SerialException as e:
            print(f"Error receiving data via radio: {e}")
            return None, None, None

# --- RadioNode Class ---

class RadioNode:
    def __init__(self, node_id, serial_port, baud_rate):
        self.node_id = node_id
        self.radio = RadioInterface(serial_port, baud_rate, node_id)
        self.peer_data = {} # Store data for other discovered peers (graphics, position, audio, nmr)
        self.peer_data_lock = threading.Lock()
        self.discovery_interval = 30 # Seconds to broadcast discovery beacons
        self.peers = {} # Store discovered peers and their last seen time

        self.interp_diff_lock = threading.Lock() # Lock for interpolation/differentiation methods

        # Start listening for incoming radio messages
        self.listen_thread = threading.Thread(target=self._radio_listen_loop, daemon=True)
        self.listen_thread.start()

        # Start broadcasting discovery beacons
        self.discovery_thread = threading.Thread(target=self._discovery_broadcast_loop, daemon=True)
        self.discovery_thread.start()

        print(f"RadioNode {self.node_id} initialized.")

    def pseudo_interpolate_arcsecant_1d_triple(self, fx, fy, fz, x_interp):
        # ... (Your existing interpolation code, no changes needed here) ...
        """
        Pseudo-interpolates 1-dimensional data with y and z datasets using an arcsecant-like function.

        Args:
            fx (numpy.ndarray): 1D array of x-data.
            fy (numpy.ndarray): 1D array of corresponding y-data.
            fz (numpy.ndarray): 1D array of corresponding z-data.
            x_interp (numpy.ndarray): Array of x-values for interpolation.

        Returns:
            tuple of numpy.ndarray: A tuple containing two arrays of interpolated y and z values.
        """
        if not (len(fx) == len(fy) == len(fz)) or len(fx) < 2:
            raise ValueError("X, Y, and Z data must have equal length and at least two points.")

        min_x, max_x = np.min(fx), np.max(fx)

        # This domain calculation for arcsecant seems unconventional and might produce NaNs
        # for abs(scaled_x) < 1. We'll keep it as per the original code structure,
        # but be aware of potential RuntimeWarnings.
        with np.errstate(invalid='ignore'): # Suppress invalid value warnings for arccos
             arcsecant_domain = np.arccos(1 / np.abs(np.linspace(-1.1, 1.1, 100)))
        # Filter out NaNs resulting from invalid arccos domain
        arcsecant_domain = arcsecant_domain[~np.isnan(arcsecant_domain)]

        if len(arcsecant_domain) < 2:
             raise ValueError("Arcsecant domain calculation resulted in insufficient valid points for scaling.")

        min_arcsecant, max_arcsecant = np.min(arcsecant_domain), np.max(arcsecant_domain)

        interp_y = []
        interp_z = []
        for x in x_interp:
            # Find the two closest points in fx
            if x <= fx[0]:
                idx1, idx2 = 0, 1
            elif x >= fx[-1]:
                idx1, idx2 = len(fx) - 2, len(fx) - 1
            else:
                # Find the index of the point immediately less than or equal to x
                idx1 = np.searchsorted(fx, x, side='right') - 1
                idx2 = idx1 + 1

            x1, x2 = fx[idx1], fx[idx2]
            y1, y2 = fy[idx1], fy[idx2]
            z1, z2 = fz[idx1], fz[idx2]

            # Scale x to the range [-1.1, 1.1] based on the original fx range
            # Avoid division by zero if max_x == min_x
            scaled_x = 2 * (x - min_x) / (max_x - min_x) - 1.1 if (max_x - min_x) != 0 else -1.1

            # Calculate the arcsecant value, suppressing invalid value warnings
            with np.errstate(invalid='ignore'):
                arcsec_val = np.arccos(1 / np.abs(scaled_x))

            # Handle potential NaN from arccos or division by zero in scaling
            if np.isnan(arcsec_val) or (max_arcsecant - min_arcsecant) == 0:
                # Fallback to linear interpolation if arcsecant scaling is problematic
                t = (x - x1) / (x2 - x1) if (x2 - x1) != 0 else 0
                interp_y.append(y1 + t * (y2 - y1))
                interp_z.append(z1 + t * (z2 - z1))
            else:
                 # Scale the linear interpolation factor using the arcsecant domain
                 scaled_t = (arcsec_val - min_arcsecant) / (max_arcsecant - min_arcsecant)
                 # Clamp scaled_t to [0, 1] in case of floating point inaccuracies
                 scaled_t = np.clip(scaled_t, 0, 1)
                 interp_y.append(y1 + (y2 - y1) * scaled_t)
                 interp_z.append(z1 + (z2 - z1) * scaled_t)


        return np.array(interp_y), np.array(interp_z)

    def pseudo_interpolate_arcsecant_nd_triple(self, all_fy_data, all_fz_data, all_fx_data, x_interp):
        # ... (Your existing interpolation code, no changes needed here) ...
        """
        Pseudo-interpolates n-dimensional data with y and z datasets.
        Performs the interpolation independently for each dimension and returns the concatenated results.

        Args:
            all_fy_data (list of numpy.ndarray): A list of y-data arrays.
            all_fz_data (list of numpy.ndarray): A list of z-data arrays.
            all_fx_data (list of numpy.ndarray): A list of corresponding x-data arrays.
            x_interp (numpy.ndarray): Array of x-values for interpolation.

        Returns:
            tuple of numpy.ndarray: A tuple containing two concatenated arrays of interpolated y and z values.
        """
        all_interp_y = []
        all_interp_z = []
        num_dimensions = len(all_fy_data)

        if not (len(all_fx_data) == num_dimensions == len(all_fz_data)):
            raise ValueError("The number of x, y, and z data arrays must match.")

        for fx, fy, fz in zip(all_fx_data, all_fy_data, all_fz_data):
            try:
                interp_y, interp_z = self.pseudo_interpolate_arcsecant_1d_triple(fx, fy, fz, x_interp)
                all_interp_y.extend(interp_y)
                all_interp_z.extend(interp_z)

            except ValueError as e:
                raise ValueError(f"ValueError during arcsecant interpolation for one dimension: {e}")
            except RuntimeWarning as e:
                 # Catch RuntimeWarnings from numpy operations within interpolation
                 print(f"RuntimeWarning during arcsecant interpolation for one dimension: {e}")
                 # Depending on desired behavior, you might re-raise or handle differently
                 # For now, we'll let it proceed but the result might contain NaNs
            except Exception as e:
                raise Exception(f"An unexpected error occurred during arcsecant interpolation for one dimension: {e}")

        return np.array(all_interp_y), np.array(all_interp_z)

    def numerical_derivative_1d(self, y_values, x_values):
        # ... (Your existing differentiation code, no changes needed here) ...
        """
        Calculates the numerical derivative of a 1D array of y-values with respect to x-values.
        Uses the central difference method where possible, and forward/backward difference at the boundaries.

        Args:
            y_values (numpy.ndarray): 1D array of y-values.
            x_values (numpy.ndarray): 1D array of corresponding x-values.

        Returns:
            numpy.ndarray: 1D array of approximate derivatives.
        """
        if len(y_values)!= len(x_values) or len(y_values) < 2:
            raise ValueError("Y and X data must have equal length and at least two points for derivative calculation.")

        derivatives = np.zeros_like(y_values, dtype=float)
        h = np.diff(x_values)

        if len(h) < 1:
            return derivatives  # Cannot compute derivative with only one point

        # Forward difference for the first point
        if h[0] != 0:
             derivatives[0] = (y_values[1] - y_values[0]) / h[0]
        else:
             derivatives[0] = 0 # Handle case where first two x points are the same

        # Central difference for interior points
        for i in range(1, len(y_values) - 1):
             # More accurate with non-uniform spacing: (f(x+h2) - f(x-h1)) / (h1 + h2)
            if h[i-1] + h[i] != 0:
                derivatives[i] = (y_values[i + 1] - y_values[i - 1]) / (h[i-1] + h[i])
            else:
                derivatives[i] = 0 # Handle case where consecutive x points are the same

        # Backward difference for the last point
        if h[-1] != 0:
            derivatives[-1] = (y_values[-1] - y_values[-2]) / h[-1]
        else:
            derivatives[-1] = 0 # Handle case where last two x points are the same


        return derivatives

    def differentiate_arcsecant_nd_triple(self, all_fy_data, all_fz_data, all_fx_data, x_eval):
        # ... (Your existing differentiation code, no changes needed here) ...
        """
        Calculates the numerical derivative of the pseudo-interpolated output (for y and z datasets).

        Args:
            all_fy_data (list of numpy.ndarray): A list of y-data arrays.
            all_fz_data (list of numpy.ndarray): A list of z-data arrays.
            all_fx_data (list of numpy.ndarray): A list of corresponding x-data arrays.
            x_eval (numpy.ndarray): Array of x-values at which to evaluate the derivative.

        Returns:
            tuple of numpy.ndarray: A tuple containing two concatenated arrays of approximate derivatives (for y and z).
        """
        num_dimensions = len(all_fy_data)
        if not (len(all_fx_data) == num_dimensions == len(all_fz_data)):
            raise ValueError("The number of x, y, and z data arrays must match.")

        all_derivatives_y = []
        all_derivatives_z = []
        for fx, fy, fz in zip(all_fx_data, all_fy_data, all_fz_data):
            try:
                if len(fx)!= len(fy) or len(fx) < 2:
                    raise ValueError("X and Y data must have equal length and at least two points for differentiation.")
                if len(fx)!= len(fz):
                    raise ValueError("X and Z data must have equal length.")
                if len(x_eval) < 2:
                    raise ValueError("At least two evaluation points are needed for differentiation.")

                # Note: Differentiating the _interpolated_ data requires the interpolated
                # points themselves. The interpolation is done at x_eval.
                # So, we interpolate first, then differentiate the resulting interpolated values
                # with respect to the evaluation points x_eval.
                interpolated_y, interpolated_z = self.pseudo_interpolate_arcsecant_1d_triple(fx, fy, fz, x_eval)

                derivatives_y = self.numerical_derivative_1d(interpolated_y, x_eval)
                derivatives_z = self.numerical_derivative_1d(interpolated_z, x_eval)

                all_derivatives_y.extend(derivatives_y)
                all_derivatives_z.extend(derivatives_z)

            except ValueError as e:
                raise ValueError(f"ValueError during differentiation for one dimension: {e}")
            except RuntimeWarning as e:
                print(f"RuntimeWarning during differentiation for one dimension: {e}")
            except Exception as e:
                raise Exception(f"An unexpected error occurred during differentiation of arcsecant interpolation for one dimension: {e}")

        return np.array(all_derivatives_y), np.array(all_derivatives_z)

    def _radio_listen_loop(self):
        """Continuously listens for incoming radio packets."""
        while True:
            try:
                dest_id, pkt_type, payload = self.radio.receive_packet()
                if dest_id is not None:
                    # If packet is for us or a broadcast (e.g., dest_id = 255)
                    if dest_id == self.node_id or dest_id == 255: # Assuming 255 is broadcast ID
                        print(f"Node {self.node_id} received packet Type {pkt_type} from radio.")
                        threading.Thread(target=self._handle_incoming_packet, args=(pkt_type, payload)).start()
            except Exception as e:
                print(f"Error in radio listen loop: {e}")
                traceback.print_exc()
            threading.Event().wait(0.01) # Small delay to prevent busy-waiting

    def _discovery_broadcast_loop(self):
        """Broadcasts a discovery beacon periodically."""
        while True:
            # Type 250: Discovery Beacon (Node ID)
            beacon_payload = struct.pack('!B', self.node_id) # Just send our ID
            self.radio.send_packet(255, 250, beacon_payload) # 255 as broadcast ID
            # print(f"Node {self.node_id} broadcasted discovery beacon.")
            threading.Event().wait(self.discovery_interval)

    def _handle_incoming_packet(self, pkt_type, payload):
        """Processes an incoming packet based on its type."""
        sender_id = struct.unpack('!B', payload[0:1])[0] # Assuming sender ID is first byte of payload
        actual_payload = payload[1:] # Rest of the payload

        # Update last seen time for sender
        self.peers[sender_id] = {'last_seen': np.datetime64('now')}

        if pkt_type == 250: # Discovery Beacon
            # Don't store data, just acknowledge presence
            # A more advanced system would send its own full initial data in response
            print(f"Node {self.node_id} received discovery beacon from Node {sender_id}.")
            return

        # For other data types, first byte of payload is usually the sender ID
        # Extract sender ID from payload to identify the source of the data
        # For simplicity, I'm assuming sender_id is prepended to actual_payload
        # This needs to be consistent with how you pack data on the sending side.

        with self.peer_data_lock:
            if sender_id not in self.peer_data:
                self.peer_data[sender_id] = {'id': sender_id}
                print(f"Discovered new peer {sender_id}.")

            peer_current_data = self.peer_data[sender_id]

            try:
                if pkt_type == 0: # Initial data from another node
                    data = self._unpack_initial_data(actual_payload)
                    peer_current_data.update(data)
                    print(f"Node {self.node_id} received initial data from peer {sender_id}.")
                    # No need to rebroadcast initial data (unless it's a new peer joining)
                    # For P2P, a new peer requests data upon discovery.

                elif pkt_type == 1: # X-axis update
                    data = self._unpack_axis_update(actual_payload)
                    peer_current_data['fx'] = data['data']
                    print(f"Node {self.node_id} received FX update from peer {sender_id}.")
                    self.broadcast_data(pkt_type, actual_payload, sender_id) # Re-broadcast to others

                elif pkt_type == 2: # Y-axis update
                    data = self._unpack_axis_update(actual_payload)
                    peer_current_data['fy'] = data['data']
                    print(f"Node {self.node_id} received FY update from peer {sender_id}.")
                    self.broadcast_data(pkt_type, actual_payload, sender_id)

                elif pkt_type == 3: # Z-axis update
                    data = self._unpack_axis_update(actual_payload)
                    peer_current_data['fz'] = data['data']
                    print(f"Node {self.node_id} received FZ update from peer {sender_id}.")
                    self.broadcast_data(pkt_type, actual_payload, sender_id)

                elif pkt_type == 4: # Eval X update
                    data = self._unpack_eval_x_update(actual_payload)
                    peer_current_data['eval_x'] = data
                    print(f"Node {self.node_id} received Eval X update from peer {sender_id}.")
                    self.broadcast_data(pkt_type, actual_payload, sender_id)

                elif pkt_type == 6: # Color update
                    data = self._unpack_color_update(actual_payload)
                    peer_current_data['color'] = data
                    print(f"Node {self.node_id} received Color update from peer {sender_id}.")
                    self.broadcast_data(pkt_type, actual_payload, sender_id)

                elif pkt_type == 7: # Position update
                    x, y, z = struct.unpack('!3f', actual_payload)
                    peer_current_data['position'] = {'x': x, 'y': y, 'z': z}
                    print(f"Node {self.node_id} received Position update from peer {sender_id}: ({x},{y},{z}).")
                    self.broadcast_data(pkt_type, actual_payload, sender_id)

                elif pkt_type == 8: # Audio update
                    data = self._unpack_audio_update(actual_payload)
                    peer_current_data['audio'] = data['audio_data']
                    print(f"Node {self.node_id} received Audio update from peer {sender_id} ({data['length']} bytes).")
                    self.broadcast_data(pkt_type, actual_payload, sender_id)

                elif pkt_type == 11: # Positional Comment
                    data = self._unpack_positional_comment(actual_payload)
                    peer_current_data['comment'] = data['comment']
                    print(f"Node {self.node_id} received Positional Comment from peer {sender_id}: '{data['comment']}'.")
                    self.broadcast_data(pkt_type, actual_payload, sender_id)

                elif pkt_type == 100: # NMR data
                    # Unpack NMR data - this logic needs to be the same as _pack_nmr_data
                    ndims = struct.unpack('!I', actual_payload[0:4])[0]
                    offset = 4
                    shape = struct.unpack(f'!{ndims}I', actual_payload[offset:offset + ndims * 4])
                    offset += ndims * 4
                    data_size = struct.unpack('!I', actual_payload[offset:offset + 4])[0]
                    offset += 4
                    nmr_bytes = actual_payload[offset:offset + data_size]

                    nmr_data = np.frombuffer(nmr_bytes, dtype=np.float32).reshape(shape)
                    peer_current_data['nmr_data'] = nmr_data
                    print(f"Node {self.node_id} received NMR data from peer {sender_id}, shape: {shape}.")
                    self.broadcast_data(pkt_type, actual_payload, sender_id) # Re-broadcast to others

                # Handle interpolation/differentiation requests (for this node to process)
                elif pkt_type == 0x00: # Request for Interpolation (original type 0)
                    print(f"Node {self.node_id} received interpolation request from {sender_id}.")
                    # Parse the data for interpolation
                    all_fx_data, all_fy_data, all_fz_data, eval_x_data = self._unpack_interp_diff_request(actual_payload)
                    with self.interp_diff_lock:
                        result_y, result_z = self.pseudo_interpolate_arcsecant_nd_triple(
                            all_fy_data, all_fz_data, all_fx_data, eval_x_data)
                    result = np.concatenate((result_y, result_z))
                    response_payload = self._pack_interp_diff_response(result)
                    self.radio.send_packet(sender_id, 0x05, response_payload) # Type 0x05 for interpolation result
                    print(f"Node {self.node_id} sent interpolation result to {sender_id}.")

                elif pkt_type == 0x01: # Request for Differentiation (original type 1)
                    print(f"Node {self.node_id} received differentiation request from {sender_id}.")
                    # Parse the data for differentiation
                    all_fx_data, all_fy_data, all_fz_data, eval_x_data = self._unpack_interp_diff_request(actual_payload)
                    with self.interp_diff_lock:
                        deriv_y, deriv_z = self.differentiate_arcsecant_nd_triple(
                            all_fy_data, all_fz_data, all_fx_data, eval_x_data)
                    result = np.concatenate((deriv_y, deriv_z))
                    response_payload = self._pack_interp_diff_response(result)
                    self.radio.send_packet(sender_id, 0x0C, response_payload) # Type 0x0C for differentiation result
                    print(f"Node {self.node_id} sent differentiation result to {sender_id}.")

                # Response types (5, 10, 12, 101, 252, 253, 254, 255) are for receiving, not storing/broadcasting in this loop
                elif pkt_type == 0x05: # Interpolation Result
                    print(f"Node {self.node_id} received interpolation result from {sender_id}.")
                    # Handle received interpolation result (e.g., update local display)
                    result = np.frombuffer(actual_payload, dtype=np.float32)
                    print(f"  Result length: {len(result)}")

                elif pkt_type == 0x0A: # Initial Data Broadcast (original type 10)
                    # This means another node is broadcasting its initial state
                    data = self._unpack_initial_data(actual_payload)
                    peer_current_data.update(data)
                    print(f"Node {self.node_id} received initial data broadcast from peer {sender_id}.")

                elif pkt_type == 0x0C: # Differentiation Result
                    print(f"Node {self.node_id} received differentiation result from {sender_id}.")
                    result = np.frombuffer(actual_payload, dtype=np.float32)
                    print(f"  Result length: {len(result)}")

                elif pkt_type == 0x65: # NMR Data Broadcast (original type 101)
                    # This means another node is broadcasting its NMR data
                    ndims = struct.unpack('!I', actual_payload[0:4])[0]
                    offset = 4
                    shape = struct.unpack(f'!{ndims}I', actual_payload[offset:offset + ndims * 4])
                    offset += ndims * 4
                    data_size = struct.unpack('!I', actual_payload[offset:offset + 4])[0]
                    offset += 4
                    nmr_bytes = actual_payload[offset:offset + data_size]

                    nmr_data = np.frombuffer(nmr_bytes, dtype=np.float32).reshape(shape)
                    peer_current_data['nmr_data'] = nmr_data
                    print(f"Node {self.node_id} received NMR data broadcast from peer {sender_id}, shape: {shape}.")

                elif pkt_type == 0xFE: # Disconnect (original type 254)
                    disconnected_id = struct.unpack('!B', actual_payload)[0] # Assuming just the ID
                    if disconnected_id in self.peer_data:
                        del self.peer_data[disconnected_id]
                        print(f"Peer {disconnected_id} disconnected.")
                    if disconnected_id in self.peers:
                        del self.peers[disconnected_id]
                    # No need to rebroadcast this; it's a notification

                elif pkt_type == 0xFF: # ID Assignment (original type 255)
                    # This type is usually for a server assigning ID to client.
                    # In P2P, IDs are pre-assigned or self-assigned.
                    # If this is used for an initial "my ID is X" message from a new peer, handle it.
                    # Assuming it means "Hey, my ID is this!"
                    assigned_id = struct.unpack('!B', actual_payload)[0]
                    print(f"Node {self.node_id} received 'My ID is' message from {sender_id}: {assigned_id}.")
                    # Update internal peer tracking, if necessary

                elif pkt_type in [252, 253]: # Server warnings/errors (now peer warnings/errors)
                    message_len = struct.unpack('!I', actual_payload[0:4])[0]
                    error_message = actual_payload[4:4+message_len].decode('utf-8')
                    print(f"Received {'ERROR' if pkt_type == 253 else 'WARNING'} from peer {sender_id}: {error_message}")

                else:
                    print(f"Node {self.node_id} received unknown packet type {pkt_type} from {sender_id}.")

            except Exception as e:
                print(f"Error processing packet type {pkt_type} from {sender_id}: {e}")
                traceback.print_exc()
                # Consider sending an error response back to sender if appropriate.

    def broadcast_data(self, pkt_type, payload, original_sender_id):
        """
        Broadcasts data to all known peers except the original sender.
        Includes original_sender_id in the payload for proper attribution.
        """
        # Prepend the original sender's ID to the payload
        full_payload = struct.pack('!B', original_sender_id) + payload

        with self.peer_data_lock:
            for peer_id in self.peers:
                if peer_id != self.node_id and peer_id != original_sender_id:
                    # We can't send to a socket directly anymore. We send via radio.
                    # Send to the specific peer_id or use a broadcast address (e.g., 255)
                    # For a true P2P broadcast, you'd send to the broadcast address.
                    # For directed messaging, you'd send to peer_id.
                    # Here, we assume a broadcast to all others.
                    success = self.radio.send_packet(255, pkt_type, full_payload) # Send to broadcast ID
                    if not success:
                        print(f"Failed to broadcast Type {pkt_type} to peer {peer_id}.")

    # --- Packing and Unpacking Helpers (Adapted for P2P Radio) ---
    # These functions now pack/unpack for radio transmission.
    # They need to handle the sender ID as part of the packet.

    def _pack_initial_data(self, data, pkt_type, sender_id):
        """Packs all initial data for broadcasting/sending to a new peer."""
        packed_data = bytearray()
        packed_data.extend(struct.pack('!B', sender_id)) # Prepend sender ID

        # Your original packing logic for initial data (fx, fy, fz, eval_x, color, position, audio, comment)
        # Ensure 'num_dimensions' is included for axis data
        num_dimensions = len(data.get('fx', []))
        packed_data.extend(struct.pack('!I', num_dimensions))

        for key_prefix in ['fx', 'fy', 'fz']:
            for arr in data.get(key_prefix, []):
                arr_bytes = arr.astype(np.float32).tobytes()
                packed_data.extend(struct.pack('!I', len(arr_bytes) // 4)) # Number of floats
                packed_data.extend(arr_bytes)

        # Eval X
        eval_x = data.get('eval_x', np.array([]))
        eval_x_bytes = eval_x.astype(np.float32).tobytes()
        packed_data.extend(struct.pack('!I', len(eval_x_bytes) // 4))
        packed_data.extend(eval_x_bytes)

        # Color (assuming 3 floats)
        color = data.get('color', (0.0, 0.0, 0.0))
        packed_data.extend(struct.pack('!3f', *color))

        # Position (assuming 3 floats)
        position = data.get('position', {'x':0.0, 'y':0.0, 'z':0.0})
        packed_data.extend(struct.pack('!3f', position['x'], position['y'], position['z']))

        # Audio data (length + bytes)
        audio_data = data.get('audio', b'')
        packed_data.extend(struct.pack('!I', len(audio_data)))
        packed_data.extend(audio_data)

        # Positional Comment (length + utf-8 bytes)
        comment = data.get('comment', '')
        comment_bytes = comment.encode('utf-8')
        packed_data.extend(struct.pack('!I', len(comment_bytes)))
        packed_data.extend(comment_bytes)

        # NMR data (optional, more complex, consider separate function or large data handling)
        nmr_data = data.get('nmr_data', None)
        if nmr_data is not None:
             ndims = nmr_data.ndim
             shape = nmr_data.shape
             data_bytes = nmr_data.astype(np.float32).tobytes()
             packed_data.extend(struct.pack('!I', ndims))
             packed_data.extend(struct.pack(f'!{ndims}I', *shape))
             packed_data.extend(struct.pack('!I', len(data_bytes)))
             packed_data.extend(data_bytes)
        else: # Indicate no NMR data
            packed_data.extend(struct.pack('!I', 0)) # 0 dimensions means no NMR data

        # Now, prepend the total length of the packed_data (excluding type and sender_id)
        # and the packet type, then the sender_id (already added)
        # This is where the actual radio packet format comes into play.
        # Let's say: PacketType (1B) | SenderID (1B) | TotalPayloadLength (4B) | Payload
        final_payload = packed_data # packed_data already has sender ID at the start
        total_payload_length = len(final_payload)
        header = struct.pack('!BI', pkt_type, total_payload_length) # PacketType, TotalPayloadLength
        return header + final_payload


    def _unpack_initial_data(self, payload):
        """Unpacks initial data received from a peer."""
        data = {}
        offset = 0

        # Assuming sender_id is handled by the packet header (or first byte of payload)
        # Let's assume the payload here *is* the data after sender ID.

        num_dimensions = struct.unpack_from('!I', payload, offset)[0]
        offset += 4

        data['fx'] = []
        data['fy'] = []
        data['fz'] = []

        for key_prefix in ['fx', 'fy', 'fz']:
            for _ in range(num_dimensions):
                num_floats = struct.unpack_from('!I', payload, offset)[0]
                offset += 4
                arr_bytes = payload[offset : offset + num_floats * 4]
                data[key_prefix].append(np.frombuffer(arr_bytes, dtype=np.float32))
                offset += num_floats * 4

        # Eval X
        num_floats = struct.unpack_from('!I', payload, offset)[0]
        offset += 4
        eval_x_bytes = payload[offset : offset + num_floats * 4]
        data['eval_x'] = np.frombuffer(eval_x_bytes, dtype=np.float32)
        offset += num_floats * 4

        # Color
        data['color'] = struct.unpack_from('!3f', payload, offset)
        offset += 3 * 4

        # Position
        pos_x, pos_y, pos_z = struct.unpack_from('!3f', payload, offset)
        data['position'] = {'x': pos_x, 'y': pos_y, 'z': pos_z}
        offset += 3 * 4

        # Audio
        audio_len = struct.unpack_from('!I', payload, offset)[0]
        offset += 4
        data['audio'] = payload[offset : offset + audio_len]
        offset += audio_len

        # Positional Comment
        comment_len = struct.unpack_from('!I', payload, offset)[0]
        offset += 4
        data['comment'] = payload[offset : offset + comment_len].decode('utf-8')
        offset += comment_len

        # NMR data
        nmr_ndims = struct.unpack_from('!I', payload, offset)[0]
        offset += 4
        if nmr_ndims > 0:
            nmr_shape = struct.unpack_from(f'!{nmr_ndims}I', payload, offset)
            offset += nmr_ndims * 4
            nmr_data_size = struct.unpack_from('!I', payload, offset)[0]
            offset += 4
            nmr_data_bytes = payload[offset : offset + nmr_data_size]
            data['nmr_data'] = np.frombuffer(nmr_data_bytes, dtype=np.float32).reshape(nmr_shape)
            offset += nmr_data_size
        else:
             data['nmr_data'] = None

        return data

    def _pack_axis_update(self, data, pkt_type, sender_id):
        """Packs axis update data."""
        packed_data = bytearray()
        packed_data.extend(struct.pack('!B', sender_id)) # Prepend sender ID
        packed_data.extend(struct.pack('!I', data['num_dimensions']))
        for arr in data['data']:
            arr_bytes = arr.astype(np.float32).tobytes()
            packed_data.extend(struct.pack('!I', len(arr_bytes) // 4)) # Number of floats
            packed_data.extend(arr_bytes)
        return struct.pack('!BI', pkt_type, len(packed_data)) + packed_data

    def _unpack_axis_update(self, payload):
        """Unpacks axis update data."""
        data = {'data': []}
        offset = 0
        # Sender ID is assumed to be handled by the outer packet
        data['num_dimensions'] = struct.unpack_from('!I', payload, offset)[0]
        offset += 4
        for _ in range(data['num_dimensions']):
            num_floats = struct.unpack_from('!I', payload, offset)[0]
            offset += 4
            arr_bytes = payload[offset : offset + num_floats * 4]
            data['data'].append(np.frombuffer(arr_bytes, dtype=np.float32))
            offset += num_floats * 4
        return data

    def _pack_eval_x_update(self, data, pkt_type, sender_id):
        packed_data = bytearray()
        packed_data.extend(struct.pack('!B', sender_id))
        data_bytes = data.astype(np.float32).tobytes()
        packed_data.extend(struct.pack('!I', len(data_bytes) // 4))
        packed_data.extend(data_bytes)
        return struct.pack('!BI', pkt_type, len(packed_data)) + packed_data

    def _unpack_eval_x_update(self, payload):
        offset = 0
        num_floats = struct.unpack_from('!I', payload, offset)[0]
        offset += 4
        data_bytes = payload[offset : offset + num_floats * 4]
        return np.frombuffer(data_bytes, dtype=np.float32)

    def _pack_color_update(self, data, pkt_type, sender_id):
        packed_data = bytearray()
        packed_data.extend(struct.pack('!B', sender_id))
        packed_data.extend(struct.pack('!3f', *data))
        return struct.pack('!BI', pkt_type, len(packed_data)) + packed_data

    def _unpack_color_update(self, payload):
        return struct.unpack_from('!3f', payload, 0)

    def _pack_audio_update(self, data, pkt_type, sender_id):
        packed_data = bytearray()
        packed_data.extend(struct.pack('!B', sender_id))
        packed_data.extend(struct.pack('!I', len(data)))
        packed_data.extend(data)
        return struct.pack('!BI', pkt_type, len(packed_data)) + packed_data

    def _unpack_audio_update(self, payload):
        offset = 0
        length = struct.unpack_from('!I', payload, offset)[0]
        offset += 4
        return {'length': length, 'audio_data': payload[offset : offset + length]}

    def _pack_positional_comment(self, data, pkt_type, sender_id):
        packed_data = bytearray()
        packed_data.extend(struct.pack('!B', sender_id))
        comment_bytes = data.encode('utf-8')
        packed_data.extend(struct.pack('!I', len(comment_bytes)))
        packed_data.extend(comment_bytes)
        return struct.pack('!BI', pkt_type, len(packed_data)) + packed_data

    def _unpack_positional_comment(self, payload):
        offset = 0
        length = struct.unpack_from('!I', payload, offset)[0]
        offset += 4
        return {'comment': payload[offset : offset + length].decode('utf-8')}

    def _pack_interp_diff_request(self, all_fx_data, all_fy_data, all_fz_data, eval_x_data, pkt_type, sender_id):
        """Packs data for interpolation/differentiation requests."""
        packed_data = bytearray()
        packed_data.extend(struct.pack('!B', sender_id)) # Prepend sender ID
        num_dimensions = len(all_fx_data)
        packed_data.extend(struct.pack('!I', num_dimensions))

        for i in range(num_dimensions):
            # Pack fx, fy, fz for each dimension
            for arr in [all_fx_data[i], all_fy_data[i], all_fz_data[i]]:
                arr_bytes = arr.astype(np.float32).tobytes()
                packed_data.extend(struct.pack('!I', len(arr_bytes) // 4)) # Number of floats
                packed_data.extend(arr_bytes)

        # Pack eval_x_data
        eval_x_bytes = eval_x_data.astype(np.float32).tobytes()
        packed_data.extend(struct.pack('!I', len(eval_x_bytes) // 4))
        packed_data.extend(eval_x_bytes)

        return struct.pack('!BI', pkt_type, len(packed_data)) + packed_data

    def _unpack_interp_diff_request(self, payload):
        """Unpacks data for interpolation/differentiation requests."""
        offset = 0
        # Sender ID is assumed to be handled by the outer packet and removed
        num_dimensions = struct.unpack_from('!I', payload, offset)[0]
        offset += 4

        all_fx_data = []
        all_fy_data = []
        all_fz_data = []

        for _ in range(num_dimensions):
            # Unpack fx
            num_floats = struct.unpack_from('!I', payload, offset)[0]
            offset += 4
            fx_bytes = payload[offset : offset + num_floats * 4]
            all_fx_data.append(np.frombuffer(fx_bytes, dtype=np.float32))
            offset += num_floats * 4

            # Unpack fy
            num_floats = struct.unpack_from('!I', payload, offset)[0]
            offset += 4
            fy_bytes = payload[offset : offset + num_floats * 4]
            all_fy_data.append(np.frombuffer(fy_bytes, dtype=np.float32))
            offset += num_floats * 4

            # Unpack fz
            num_floats = struct.unpack_from('!I', payload, offset)[0]
            offset += 4
            fz_bytes = payload[offset : offset + num_floats * 4]
            all_fz_data.append(np.frombuffer(fz_bytes, dtype=np.float32))
            offset += num_floats * 4

        # Unpack eval_x_data
        eval_x_count = struct.unpack_from('!I', payload, offset)[0]
        offset += 4
        eval_x_bytes = payload[offset : offset + eval_x_count * 4]
        eval_x_data = np.frombuffer(eval_x_bytes, dtype=np.float32)
        offset += eval_x_count * 4

        return all_fx_data, all_fy_data, all_fz_data, eval_x_data

    def _pack_interp_diff_response(self, result_array):
        """Packs interpolation/differentiation results."""
        result_bytes = result_array.astype(np.float32).tobytes()
        return struct.pack('!I', len(result_bytes)) + result_bytes # Just length + data

    def _pack_nmr_data(self, nmr_data, pkt_type, sender_id):
        packed_data = bytearray()
        packed_data.extend(struct.pack('!B', sender_id)) # Prepend sender ID

        ndims = nmr_data.ndim
        shape = nmr_data.shape
        data_bytes = nmr_data.astype(np.float32).tobytes()

        packed_data.extend(struct.pack('!I', ndims))
        packed_data.extend(struct.pack(f'!{ndims}I', *shape))
        packed_data.extend(struct.pack('!I', len(data_bytes)))
        packed_data.extend(data_bytes)
        return struct.pack('!BI', pkt_type, len(packed_data)) + packed_data

    # You'll also need a way for the "client" part to send requests:
    def request_interpolation(self, destination_id, all_fx, all_fy, all_fz, eval_x):
        # Pack the request using your new radio-specific packing
        payload = self._pack_interp_diff_request(all_fx, all_fy, all_fz, eval_x, 0x00, self.node_id)
        self.radio.send_packet(destination_id, 0x00, payload)

    def request_differentiation(self, destination_id, all_fx, all_fy, all_fz, eval_x):
        payload = self._pack_interp_diff_request(all_fx, all_fy, all_fz, eval_x, 0x01, self.node_id)
        self.radio.send_packet(destination_id, 0x01, payload)

    def send_nmr_data(self, destination_id, nmr_data):
        payload = self._pack_nmr_data(nmr_data, 100, self.node_id)
        self.radio.send_packet(destination_id, 100, payload)

    def send_position_update(self, destination_id, x, y, z):
        payload = struct.pack('!B3f', self.node_id, x, y, z)
        self.radio.send_packet(destination_id, 7, payload)

# Example Usage (Conceptual)
if __name__ == "__main__":
    # You'll need to know your serial port and choose unique IDs for each node.
    # For Windows: 'COMx', For Linux: '/dev/ttyUSBx' or '/dev/ttyACMx'
    try:
        # Replace with your actual serial port and baud rate
        node1 = RadioNode(node_id=1, serial_port='/dev/ttyUSB0', baud_rate=9600)
        node2 = RadioNode(node_id=2, serial_port='/dev/ttyUSB1', baud_rate=9600)

        # Simulate some data
        fx_data = [np.array([0., 1., 2.]), np.array([10., 11., 12.])]
        fy_data = [np.array([0., 1., 4.]), np.array([100., 121., 144.])]
        fz_data = [np.array([0., 2., 8.]), np.array([200., 242., 288.])]
        eval_x = np.array([0.5, 1.5, 10.5, 11.5])
        dummy_nmr_data = np.random.rand(16, 16).astype(np.float32)

        import time
        time.sleep(5) # Give nodes time to start and discover each other

        print("\n--- Node 1 sending data to Node 2 ---")
        # Node 1 sends interpolation request to Node 2
        node1.request_interpolation(2, fx_data, fy_data, fz_data, eval_x)
        time.sleep(1)
        node1.send_nmr_data(2, dummy_nmr_data)
        time.sleep(1)
        node1.send_position_update(2, 1.0, 2.0, 3.0)

        print("\n--- Node 2 sending data to Node 1 ---")
        # Node 2 sends differentiation request to Node 1
        node2.request_differentiation(1, fx_data, fy_data, fz_data, eval_x)
        time.sleep(1)
        node2.send_position_update(1, 4.0, 5.0, 6.0)

        # Keep the main threads alive to allow background threads to run
        while True:
            time.sleep(1)

    except serial.SerialException as se:
        print(f"Serial Port Error: {se}. Make sure your serial ports are correct and not in use.")
        print("Please check your serial port configuration.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
