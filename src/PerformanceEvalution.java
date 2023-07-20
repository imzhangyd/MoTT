import javax.swing.JFrame;
import javax.swing.JPanel;

/**
 *
 * Software for computing tracking performance.
 * This software was originally developed for the ISBI'2012 Particle Tracking Challenge.
 * Read the manual in README.txt before using it.
 * 
 * Several tracking criteria are computed based on an optimal pairing between a set of reference tracks
 * and a set of candidate tracks.
 * 
 * The software can be used for batch computation by specifying the options and input/output files as the
 * arguments of the program.
 * 
 * A GUI can also be displayed for setting up the computation if no input argument is specified.
 * 
 * @version February 5, 2012
 * 
 * @author Nicolas Chenouard
 *
 * */

public class PerformanceEvalution
{
	public static void main(String [ ] args)
	{	
		if (args.length==0)
		{
			//launch the GUI
			JPanel panel = new TrackingPerformancePanel();
			JFrame frame = new JFrame();
			frame.setContentPane(panel);
			frame.setTitle("Tracking performance measures");
			frame.pack();
			frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
			frame.setVisible(true);
			return;
		}
		else
		{
			// batch computation mode
			BatchPerformanceEvaluation evaluator = new BatchPerformanceEvaluation();
			try{
				boolean successInit = evaluator.init(args);
				if (!successInit)
				{
					System.out.println(evaluator.getInitLog());
					displayHelp();
					return;
				}
				else
				{
					if (!evaluator.getInitLog().isEmpty())
						System.out.println(evaluator.getInitLog());
					boolean successPairing = evaluator.pairTracks();
					if (!successPairing)
					{
						System.out.println(evaluator.getPairingLog());
						return;
					}
					if (!evaluator.getPairingLog().isEmpty())
						System.out.println(evaluator.getPairingLog());
					boolean successExport = evaluator.outputResults();
					if (!successExport)
					{
						System.out.println(evaluator.getExportLog());
						return;
					}
				}
			}
			catch (Exception e)
			{
				displayHelp();
			}				
		}
	}

	/**
	 * Display the manual of the software in the standard output stream
	 * */
	public static void displayHelp()
	{
		System.out.println(BatchPerformanceEvaluation.getHelp());
	}
}
