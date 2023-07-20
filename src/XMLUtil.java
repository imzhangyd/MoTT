
import java.io.File;
import java.util.ArrayList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.w3c.dom.Attr;
import org.w3c.dom.Document;
import org.w3c.dom.DocumentType;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

/** These are some function copied from XMLUtil and FileUtil from the ICY software
 * check http://bioimageanalysis.org/icy for more information about icy
 * 
 * @author Fabrice de Chaumont
 * @author Stephane Dallongeville
 * 
 * modified by Nicolas Chenouard for the purpose of this software development
 * */

public class XMLUtil {

	public static final String NODE_ROOT_NAME = "root";

	private static DocumentBuilderFactory documentBuilderFactory = null;
	private static TransformerFactory transformerFactory = null;

	/**
	 * Load XML Document from specified file.<br>
	 * Return null if no document can be loaded.
	 */
	public static Document loadDocument(File f, boolean showError)
	{
		if ((f == null) || !f.exists())
		{
			if (showError)
				System.err.println("XMLUtil.loadDocument('" + f + "') error : file not found !");

			return null;
		}
		final DocumentBuilder docBuilder = getDocBuilder();
		if (docBuilder != null)
		{
			try
			{
				return docBuilder.parse(f);
			}
			catch (Exception e)
			{
				if (showError)
				{
					System.err.println("XMLUtil.loadDocument('" + f.getPath() + "') error :");
					e.printStackTrace();
				}
			}
		}
		return null;
	}

	/**
	 * Return the root element for specified document<br>
	 * Create if it does not already exist with the specified name
	 */
	private static Element getRootElement(Document doc, boolean create, String name)
	{
		if (doc != null)
		{
			Element result = doc.getDocumentElement();

			if ((result == null) && create)
			{
				result = doc.createElement(name);
				doc.appendChild(result);
			}

			return result;
		}
		return null;
	}

	/**
	 * Return the root element for specified document<br>
	 * Create if it does not already exist with the default {@link #NODE_ROOT_NAME}
	 */
	public static Element getRootElement(Document doc, boolean create)
	{
		return getRootElement(doc, create, NODE_ROOT_NAME);
	}

	/**
	 * Return the root element for specified document (null if not found)<br>
	 */
	public static Element getRootElement(Document doc)
	{
		return getRootElement(doc, false);
	}

	/**
	 * Get all child element of specified node
	 */
	public static ArrayList<Element> getSubElements(Node node)
	{
		final ArrayList<Element> result = new ArrayList<Element>();
		final NodeList nodeList = node.getChildNodes();

		// have to catch exception as sometime NodeList launch null pointer exception
		try
		{
			if (nodeList != null)
			{
				for (int i = 0; i < nodeList.getLength(); i++)
				{
					final Node subNode = nodeList.item(i);

					if (subNode instanceof Element)
						result.add((Element) subNode);
				}
			}
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}

		return result;
	}
	
	/**
	 * Get all child element with specified name of specified node
	 */
	public static ArrayList<Element> getSubElements(Node node, String name)
	{
		final ArrayList<Element> result = new ArrayList<Element>();
		final NodeList nodeList = node.getChildNodes();

		// have to catch exception as sometime NodeList launch null pointer exception
		try
		{
			if (nodeList != null)
			{
				for (int i = 0; i < nodeList.getLength(); i++)
				{
					final Node subNode = nodeList.item(i);

					if (subNode.getNodeName().equals(name) && (subNode instanceof Element))
						result.add((Element) subNode);
				}
			}
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}

		return result;
	}

	/**
	 * Get the first child element with specified name from node.<br>
	 * Return null if not found.
	 */
	public static Element getSubElement(Node node, String name)
	{
		final NodeList nodeList = node.getChildNodes();

		// have to catch exception as sometime NodeList launch null pointer exception
		try
		{
			if (nodeList != null)
			{
				for (int i = 0; i < nodeList.getLength(); i++)
				{
					final Node subNode = nodeList.item(i);

					if (subNode.getNodeName().equals(name) && (subNode instanceof Element))
						return (Element) subNode;
				}
			}
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
		return null;
	}

	private static synchronized DocumentBuilder getDocBuilder()
	{
		init();
		try
		{
			return documentBuilderFactory.newDocumentBuilder();
		}
		catch (ParserConfigurationException e)
		{
			return null;
		}
	}

	private static synchronized void init()
	{
		if (documentBuilderFactory == null)
			documentBuilderFactory = DocumentBuilderFactory.newInstance();

		if (transformerFactory == null)
		{
			transformerFactory = TransformerFactory.newInstance();
			transformerFactory.setAttribute("indent-number", new Integer(4));
		}
	}

	/**
	 * Get attribute value as integer from the specified Element.<br>
	 * If no attribute found 'def' value is returned.
	 */
	public static int getAttributeIntValue(Element element, String attribute, int def)
	{
		return getInt(getAttributeValue(element, attribute, ""), def);
	}

	/**
	 * Get attribute value from the specified Element.<br>
	 * If no attribute found 'def' value is returned.
	 */
	public static String getAttributeValue(Element element, String attribute, String def)
	{
		if (element != null)
		{
			final Attr attr = element.getAttributeNode(attribute);

			if (attr != null)
				return attr.getValue();
		}
		return def;
	}

	private static int getInt(String value, int def)
	{
		try
		{
			return Integer.parseInt(value);
		}
		catch (NumberFormatException E)
		{
			return def;
		}
	}

	/**
	 * Get attribute value as double from the specified Element.<br>
	 * If no attribute found 'def' value is returned.
	 */
	public static double getAttributeDoubleValue(Element element, String attribute, double def)
	{
		return getDouble(getAttributeValue(element, attribute, ""), def);
	}

	private static double getDouble(String value, double def)
	{
		try
		{
			return Double.parseDouble(value);
		}
		catch (NumberFormatException E)
		{
			return def;
		}
	}

	/**
	 * Create and return an empty XML Document.
	 */
	public static Document createDocument(boolean createRoot)
	{
		final DocumentBuilder docBuilder = getDocBuilder();

		if (docBuilder != null)
		{
			// create document
			final Document result = docBuilder.newDocument();

			// add default "root" element if wanted
			if (createRoot)
				createRootElement(result);

			return result;
		}
		return null;
	}

	/**
	 * Create root element for specified document if it does not already exist and return it
	 */
	public static Element createRootElement(Document doc)
	{
		return createRootElement(doc, NODE_ROOT_NAME);
	}

	/**
	 * Create root element for specified document if it does not already exist and return it
	 */
	public static Element createRootElement(Document doc, String name)
	{
		return getRootElement(doc, true, name);
	}

	/**
	 * Set an attribute and his value as double to the specified node
	 */
	public static void setAttributeDoubleValue(Element element, String attribute, double value)
	{
		setAttributeValue(element, attribute, Double.toString(value));
	}

	/**
	 * Set an attribute and his value to the specified node
	 */
	public static void setAttributeValue(Element element, String attribute, String value)
	{
		element.setAttribute(attribute, value);
	}

	/**
	 * Set an attribute and his value as integer to the specified node
	 */
	public static void setAttributeIntValue(Element element, String attribute, int value)
	{
		setAttributeValue(element, attribute, Integer.toString(value));
	}

	/**
	 * Save the specified XML Document to specified filename.<br>
	 * Return false if an error occurred.
	 */
	public static boolean saveDocument(Document doc, String filename)
	{
		return saveDocument(doc, createFile(filename));
	}

	private static synchronized Transformer getTransformer()
	{
		init();
		try
		{
			final Transformer result = transformerFactory.newTransformer();

			result.setOutputProperty(OutputKeys.METHOD, "xml");
			result.setOutputProperty(OutputKeys.ENCODING, "ISO-8859-1");
			// result.setOutputProperty(OutputKeys.ENCODING, "UTF-8");
			result.setOutputProperty(OutputKeys.OMIT_XML_DECLARATION, "no");
			result.setOutputProperty(OutputKeys.INDENT, "yes");

			return result;
		}
		catch (TransformerConfigurationException e)
		{
			return null;
		}
	}

	/**
	 * Save the specified XML Document to specified file.<br>
	 * Return false if an error occurred.
	 */
	public static boolean saveDocument(Document doc, File f)
	{
		final Transformer transformer = getTransformer();

		if (transformer != null)
		{
			final DocumentType doctype = doc.getDoctype();

			if (doctype != null)
			{
				transformer.setOutputProperty(OutputKeys.DOCTYPE_PUBLIC, doctype.getPublicId());
				transformer.setOutputProperty(OutputKeys.DOCTYPE_SYSTEM, doctype.getSystemId());
			}

			doc.normalizeDocument();

			try
			{
				transformer.transform(new DOMSource(doc), new StreamResult(f));
			}
			catch (Exception e)
			{
				return false;
			}

			return true;
		}
		return false;
	}

	public static File createFile(String filename)
	{
		return createFile(new File(getGenericPath(filename)));
	}

	public static File createFile(File file)
	{
		if (!file.exists())
		{
			// create parent directory if not exist
			ensureParentDirExist(file);

			try
			{
				file.createNewFile();
			}
			catch (Exception e)
			{
				e.printStackTrace();
				return null;
			}
		}
		return file;
	}

	public static String getGenericPath(String path)
	{
		if (path != null)
			return path.replace('\\', '/');
		return null;
	}

	public static boolean ensureParentDirExist(File file)
	{
		final String dir = file.getParent();

		if (dir != null)
			return createDir(dir);
		return true;
	}

	public static boolean createDir(String dirname)
	{
		return createDir(new File(getGenericPath(dirname)));
	}

	public static boolean createDir(File dir)
	{
		if (!dir.exists())
			return dir.mkdirs();
		return true;
	}
}
